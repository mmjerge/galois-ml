"""Trusted release runtime: the enforcement layer around the gate.

`galois_ml.gate` computes a gate DECISION; this module manages the
lifecycle that makes the decision binding: freeze/hash candidates,
certify against a private fresh holdout, atomically release the
selected artifact or the fallback.

IMPLEMENTATION STATUS (kept honest after two adversarial source
audits). What is implemented and tested here:

* freeze-and-hash of candidate SOURCE before certification randomness
  exists; submission after freeze raises; hashes re-verified both at
  evaluation and at release;
* a locked, exception-safe lifecycle: OPEN -> FROZEN -> CERTIFYING ->
  CERTIFIED -> RELEASED, with any uncaught certification failure
  entering a terminal FAILED state (no retry against a half-used
  holdout, no mixed-holdout decision lists);
* release authorization from PRIVATE IMMUTABLE records captured at
  certification commit -- the GateDecision objects handed to callers
  are deep copies, so mutating them cannot alter what release() does;
* bounded, non-executable IPC: the candidate process returns a fixed
  header plus exactly m little-endian int64 labels as raw bytes,
  received with a size cap and deadline; the trusted parent never
  unpickles worker-supplied objects;
* label-alphabet validation of candidate AND fallback predictions
  (integers drawn from the labels observed in the holdout);
* a private row permutation: the child sees features in an order
  unrelated to generation order, blunting positional label leakage
  from any generator structure;
* the released candidate predictor executes through the SAME isolated
  protocol as certification (spawned process, byte protocol, timeout,
  validation); a deployment-time failure raises -- it does not
  silently substitute anything, because a post-release substitution
  would falsify the audit record;
* worker cleanup: terminate, then kill; startup failures and malformed
  or oversized responses all map to fallback with the reason recorded.

What is NOT established here, and must be supplied by the surrounding
system before "enforced gated release" is claimed end to end:

* semantic artifact identity: only source bytes are hashed; the
  executor, interpreter, and dependency environment are part of the
  artifact and need a manifest and restriction, not a hash of the
  world;
* `run_predict_code`'s own restrictions (imports, filesystem, network)
  and POINTWISE prediction semantics -- the Hoeffding argument assumes
  per-row losses from a fixed predictor, and a batch program with
  cross-row state is outside it;
* child resource limits beyond wall clock (memory, CPU, descendants);
* a protected deployment sink that accepts only runtime-authorized
  artifacts;
* durable, authenticated audit storage (the audit dict is a local
  diagnostic; sign it or store it append-only for release evidence);
* crash durability across process restarts (single-use here is a
  Python-object property).

The fallback executes in the trusted parent, unsandboxed: it is a
trusted component by assumption, and its outputs are validated but its
execution is not confined.
"""

from __future__ import annotations

import copy
import hashlib
import multiprocessing as mp
import secrets
import struct
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from galois_ml.gate import GateDecision, certify_release

#: Lifecycle states. Transitions only move rightward; FAILED is terminal.
OPEN, FROZEN, CERTIFYING, CERTIFIED, RELEASED, FAILED = (
    "open", "frozen", "certifying", "certified", "released", "failed"
)

_MAGIC_OK = b"GLOK"
_MAGIC_ERR = b"GLER"
_MAX_ERR_BYTES = 2000


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _candidate_worker(code: str, X: np.ndarray, conn) -> None:
    """Runs in a spawned child. Receives features only. Replies with a
    bounded byte message, never a pickled object: 'GLOK' + m int64
    labels, or 'GLER' + a truncated UTF-8 reason."""
    try:
        from galois_ml.llm_eval import run_predict_code

        preds = np.asarray(run_predict_code(code, X))
        if preds.shape != (len(X),):
            raise ValueError(f"invalid output shape {preds.shape}")
        as_int = preds.astype(np.int64)
        if not np.array_equal(as_int, preds.astype(float)):
            raise ValueError("non-integer predictions")
        conn.send_bytes(_MAGIC_OK + as_int.astype("<i8").tobytes())
    except BaseException as exc:  # noqa: BLE001 - report, parent decides
        msg = f"{type(exc).__name__}: {exc}".encode(errors="replace")
        try:
            conn.send_bytes(_MAGIC_ERR + msg[:_MAX_ERR_BYTES])
        except Exception:
            pass
    finally:
        conn.close()


def isolated_predict(
    code: str, X: np.ndarray, timeout_s: float = 60.0
) -> np.ndarray:
    """Execute candidate code on features in a spawned process with the
    bounded byte protocol -- the SAME path certification uses. Raises
    on any failure; used for released candidate artifacts, where a
    silent substitution would falsify the audit record."""
    preds, note = _run_isolated(code, np.asarray(X, dtype=float), timeout_s)
    if preds is None:
        raise RuntimeError(f"released-artifact execution failed: {note}")
    return preds


def _run_isolated(
    code: str, X: np.ndarray, timeout_s: float
) -> tuple[np.ndarray | None, str]:
    """One isolated evaluation. Every failure mode maps to (None,
    reason): startup failure, timeout, death, oversized or malformed
    reply, worker-reported error."""
    m = len(X)
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    try:
        proc = ctx.Process(target=_candidate_worker, args=(code, X, child))
        proc.start()
    except Exception as exc:
        parent.close()
        child.close()
        return None, f"worker startup failed: {type(exc).__name__}: {exc}"
    child.close()
    max_len = len(_MAGIC_OK) + 8 * m + _MAX_ERR_BYTES
    try:
        if not parent.poll(timeout_s):
            return None, "timeout"
        raw = parent.recv_bytes(maxlength=max_len)
    except (EOFError, OSError) as exc:
        return None, f"candidate process died without a result "\
                     f"({type(exc).__name__})"
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join(5)
        parent.close()
    if raw[:4] == _MAGIC_ERR:
        return None, "candidate raised: " + raw[4:].decode(errors="replace")
    if raw[:4] != _MAGIC_OK or len(raw) != 4 + 8 * m:
        return None, f"malformed worker reply ({len(raw)} bytes)"
    preds = np.frombuffer(raw[4:], dtype="<i8").copy()
    return preds, "ok"


@dataclass(frozen=True)
class _Authorization:
    """Private immutable release authorization, captured at
    certification commit. release() consults ONLY these records, never
    the (deep-copied) GateDecision objects callers can see."""

    index: int
    accepted: bool
    paired_loss_difference: float
    artifact_hash: str


@dataclass
class ReleasedArtifact:
    kind: str                    # 'candidate' | 'fallback'
    artifact_hash: str | None    # sha256 of certified code (candidates)
    decision: GateDecision       # deep copy
    audit: dict                  # deep copy; includes 'released' fields
    predict: Callable[[np.ndarray], np.ndarray]


class ReleaseRuntime:
    """One-shot certification-and-release lifecycle. Single-use by
    construction (within this process); see module docstring for the
    obligations discharged here versus those the surrounding system
    must supply."""

    def __init__(
        self,
        task: str,
        fallback_predict: Callable[[np.ndarray], np.ndarray],
        m: int = 2000,
        n_coeffs: int = 1000,
        epsilon: float = 0.02,
        delta: float = 0.05,
        eval_timeout_s: float = 60.0,
    ):
        # validate configuration before any state exists
        if not (np.isfinite(epsilon) and 0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon must be finite and in [0, 1]")
        if not (np.isfinite(delta) and 0.0 < delta < 1.0):
            raise ValueError("delta must be finite and in (0, 1)")
        if not isinstance(m, int) or m <= 0:
            raise ValueError("m must be a positive integer")
        if eval_timeout_s <= 0:
            raise ValueError("eval_timeout_s must be positive")
        self._task = task
        self._fallback = fallback_predict
        self._m = m
        self._n_coeffs = n_coeffs
        self._epsilon = epsilon
        self._delta = delta
        self._timeout = eval_timeout_s
        self._lock = threading.Lock()
        self._state = OPEN
        self._candidates: list[str] = []
        self._hashes: list[str] = []
        self._decisions: list[GateDecision] = []
        self._auth: tuple[_Authorization, ...] = ()
        self._eval_notes: list[str] = []
        self._seed: int | None = None
        self._fallback_holdout_acc: float | None = None

    # -- lifecycle ----------------------------------------------------

    def submit(self, candidate_code: str) -> str:
        with self._lock:
            if self._state != OPEN:
                raise RuntimeError(
                    f"cannot submit in state {self._state!r}: candidates "
                    f"are frozen before certification randomness exists"
                )
            code = str(candidate_code)
            self._candidates.append(code)
            h = _sha256(code)
            self._hashes.append(h)
            return h

    def freeze(self) -> list[str]:
        with self._lock:
            if self._state != OPEN:
                raise RuntimeError(f"cannot freeze in state {self._state!r}")
            if not self._candidates:
                raise RuntimeError("no candidates submitted")
            self._state = FROZEN
            return list(self._hashes)

    def certify(self) -> list[GateDecision]:
        """Draw the private holdout and gate every frozen candidate at
        delta / K. Single use; any uncaught failure is terminal
        (FAILED), so a retry cannot mix holdouts or decision lists.
        Returns DEEP COPIES: mutating them cannot alter release()."""
        with self._lock:
            if self._state != FROZEN:
                raise RuntimeError(
                    f"cannot certify in state {self._state!r}: freeze "
                    f"first, certify once"
                )
            self._state = CERTIFYING
        try:
            decisions, notes, seed, fb_acc = self._certify_body()
        except BaseException:
            with self._lock:
                self._state = FAILED
            raise
        with self._lock:
            self._decisions = decisions
            self._eval_notes = notes
            self._seed = seed
            self._fallback_holdout_acc = fb_acc
            self._auth = tuple(
                _Authorization(
                    index=i,
                    accepted=(d.released == "candidate"),
                    paired_loss_difference=d.paired_loss_difference,
                    artifact_hash=self._hashes[i],
                )
                for i, d in enumerate(decisions)
            )
            self._state = CERTIFIED
        return copy.deepcopy(decisions)

    def _certify_body(self):
        """All certification work, accumulated in locals and committed
        by the caller in one step."""
        from galois_ml.data import make_dataset

        seed = secrets.randbits(63)
        X, y, _ = make_dataset(self._task, self._m, self._n_coeffs, seed=seed)
        alphabet = np.unique(y)
        fb_preds = np.asarray(self._fallback(X))
        if fb_preds.shape != y.shape or not np.all(np.isin(fb_preds,
                                                           alphabet)):
            raise RuntimeError(
                "trusted fallback returned invalid predictions; refusing "
                "to certify (trusted-component failure)"
            )
        # private permutation: the child never sees generation order, so
        # positional structure in the generator cannot act as a label
        # side channel
        perm = np.random.default_rng(secrets.randbits(63)).permutation(
            self._m
        )
        X_shuffled = X[perm]
        decisions, notes = [], []
        K = len(self._candidates)
        for code, expected_hash in zip(self._candidates, self._hashes):
            if _sha256(code) != expected_hash:
                raise RuntimeError("frozen artifact hash mismatch")
            preds_shuf, note = _run_isolated(code, X_shuffled, self._timeout)
            preds = None
            if preds_shuf is not None:
                unshuf = np.empty(self._m, dtype=np.int64)
                unshuf[perm] = preds_shuf
                if np.all(np.isin(unshuf, alphabet)):
                    preds = unshuf
                else:
                    note = "predictions outside the label alphabet"
            notes.append(note)
            decisions.append(certify_release(
                preds, fb_preds, y,
                epsilon=self._epsilon, delta=self._delta, n_candidates=K,
                fallback_kind=getattr(self._fallback, "fallback_kind", None),
            ))
        fb_acc = float(np.mean(fb_preds == y))
        return decisions, notes, seed, fb_acc

    def release(self) -> ReleasedArtifact:
        """Atomic, single-use selection from the PRIVATE immutable
        authorization records. The only path that yields an executable
        artifact; released candidates execute through the same isolated
        protocol used at certification."""
        with self._lock:
            if self._state != CERTIFIED:
                raise RuntimeError(
                    f"cannot release in state {self._state!r}: certify "
                    f"first, release once"
                )
            self._state = RELEASED
            auth = self._auth

        accepted = [(a.paired_loss_difference, a.index)
                    for a in auth if a.accepted]
        audit = {
            "task": self._task,
            "holdout_seed": self._seed,  # disclosed only now
            "holdout_m": self._m,
            "epsilon": self._epsilon,
            "delta_family": self._delta,
            "n_candidates": len(self._candidates),
            "candidate_hashes": list(self._hashes),
            "evaluation_notes": list(self._eval_notes),
            "fallback_holdout_accuracy": self._fallback_holdout_acc,
            "decisions": [d.to_dict() for d in self._decisions],
            "released_at": time.time(),
        }
        if accepted:
            _, idx = min(accepted)
            code = self._candidates[idx]
            # hash re-verified at release as well as at evaluation
            if _sha256(code) != auth[idx].artifact_hash:
                with self._lock:
                    self._state = FAILED
                raise RuntimeError("artifact hash mismatch at release")
            audit["released"] = "candidate"
            audit["released_artifact_hash"] = auth[idx].artifact_hash
            timeout = self._timeout
            return ReleasedArtifact(
                kind="candidate",
                artifact_hash=auth[idx].artifact_hash,
                decision=copy.deepcopy(self._decisions[idx]),
                audit=copy.deepcopy(audit),
                predict=lambda X, _c=code, _t=timeout: isolated_predict(
                    _c, X, _t
                ),
            )
        audit["released"] = "fallback"
        audit["released_artifact_hash"] = None
        worst = max(
            self._decisions,
            key=lambda d: (d.paired_loss_difference
                           if np.isfinite(d.paired_loss_difference)
                           else np.inf),
        )
        return ReleasedArtifact(
            kind="fallback",
            artifact_hash=None,
            decision=copy.deepcopy(worst),
            audit=copy.deepcopy(audit),
            predict=self._fallback,
        )
