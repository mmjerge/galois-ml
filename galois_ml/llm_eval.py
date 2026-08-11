"""LLM / agent evaluation harness for the Galois benchmarks.

Protocols (all scored against provable ground truth):

* rule_induction - the model sees the task description, a data excerpt,
  and empirical per-class value distributions of prime-power features
  (the same evidence a human interpreter would use). It must return a
  JSON object with a natural-language `conjecture` and a Python
  `predict(zeta) -> label` function. We execute the function on the
  held-out test set in a subprocess and report accuracy and regret
  against the Bayes ceiling. The conjecture is stored for grading
  against the known theorem.

* derivation - same output contract, but NO data is shown: only the
  mathematical setup (Euler factor formula, candidate groups). Tests
  pure reasoning. Use held-out tasks (degree14/21/25) to control for
  training-data contamination, since the paper's criteria are public.

* direct_classification - few-shot in-context classification on a
  batch of test rows; the model returns a JSON list of labels.

Models are provider-agnostic: any callable `prompt: str -> response:
str`. A Bedrock adapter is included. Offline validation uses an oracle
(theorem-as-code, should approach the ceiling) and a majority-class
floor, so the harness is verified before any API call is made.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass

import numpy as np

from galois_ml.benchmark import GaloisBenchmark
from galois_ml.groups import GROUPS
from galois_ml.zeta import _factor_table, primes_up_to

CODE_TIMEOUT_SECONDS = 60


@dataclass
class EvalResult:
    task: str
    protocol: str
    model_name: str
    valid_response: bool
    accuracy: float | None
    bayes_accuracy: float
    regret: float | None
    conjecture: str | None
    n_test: int
    error: str | None = None
    raw_response: str | None = None  # truncated; for auditing

    def to_dict(self) -> dict:
        return asdict(self)


# -- prompt construction ---------------------------------------------------

_MATH_SETUP = """\
Setting: K/Q is a Galois number field of degree {degree}. Its Dedekind
zeta function has coefficients a_1, a_2, a_3, ... where a_n counts the
ideals of norm n in the ring of integers of K. You are given the vector
zeta = [a_1, ..., a_{n_coeffs}] (so zeta[i] is a_(i+1), 0-indexed).

Key facts you may use:
- The coefficients are multiplicative: a_(mn) = a_m * a_n for coprime m, n.
- For a prime p with decomposition type (e, f, g) in K (f = inertia
  degree, g = number of primes above p, efg = {degree}), the p-power
  coefficients are: a_(p^k) = C(g + k/f - 1, g - 1) if f divides k,
  and 0 otherwise. In particular a_(p^f) = g and a_(p^k) = 0 for
  0 < k < f.
- For all but finitely many primes e = 1, so f*g = {degree}, and by the
  Chebotarev density theorem f is distributed as the order of a
  uniformly random element of the Galois group.
- Which f can occur (and how often) is constrained by the group:
  f must be the order of some element of Gal(K/Q).

The Galois group of K is one of: {classes}.
Candidate group element-order counts (order: number of elements):
{order_tables}
"""

_OUTPUT_CONTRACT = """\
Respond with ONLY a JSON object (no other text) of the form:
{{
  "conjecture": "<one or two sentences stating your classification rule
                 as a mathematical criterion>",
  "python": "def predict(zeta):\\n    ..."
}}
The python string must define a function predict(zeta) taking the list
of {n_coeffs} floats and returning the integer label: {label_map}.
It must be self-contained (standard library only).
"""


def _order_tables(class_names: list[str]) -> str:
    lines = []
    for name in class_names:
        g = GROUPS[name]
        orders = ", ".join(
            f"{o}: {c}" for o, c in sorted(g.element_orders.items())
        )
        lines.append(f"  {name} (order {g.order}): {{{orders}}}")
    return "\n".join(lines)


def _label_map(class_names: list[str]) -> str:
    return ", ".join(f"{i} = {n}" for i, n in enumerate(class_names))


def prime_power_summary(benchmark: GaloisBenchmark, max_lines: int = 40) -> str:
    """Empirical per-class value distributions of prime-power features
    in the TRAINING split - the evidence a human interpreter would
    tabulate (compare Tables 16-27 of the paper). Computed from data,
    not from theory: discovering the structure is the model's job."""
    n_coeffs = benchmark.X_train.shape[1]
    _, decomp = _factor_table(n_coeffs)
    prime_powers = []
    for n in range(2, n_coeffs + 1):
        m, p, _ = decomp[n]
        if m == 1:  # n is a pure prime power
            prime_powers.append(n)
    lines = []
    for n in prime_powers[:max_lines]:
        parts = []
        for gi, cls in enumerate(benchmark.class_names):
            vals = benchmark.X_train[benchmark.y_train == gi, n - 1]
            uniq, counts = np.unique(vals, return_counts=True)
            freq = ", ".join(
                f"{int(v)}: {c / len(vals):.2f}" for v, c in zip(uniq, counts)
            )
            parts.append(f"{cls} {{{freq}}}")
        lines.append(f"  a_{n}: " + " | ".join(parts))
    return "\n".join(lines)


def data_excerpt(
    benchmark: GaloisBenchmark, n_examples: int = 4, seed: int = 0
) -> str:
    """A few labeled training rows (balanced across classes)."""
    rng = np.random.default_rng(seed)
    lines = []
    for gi, cls in enumerate(benchmark.class_names):
        idx = np.where(benchmark.y_train == gi)[0]
        for i in rng.choice(idx, size=n_examples, replace=False):
            row = ", ".join(str(int(v)) for v in benchmark.X_train[i])
            lines.append(f"  label {gi} ({cls}): [{row}]")
    return "\n".join(lines)


def build_prompt(
    benchmark: GaloisBenchmark,
    protocol: str = "rule_induction",
    n_examples: int = 3,
    seed: int = 0,
) -> str:
    """Assemble the evaluation prompt for a benchmark task."""
    degree = GROUPS[benchmark.class_names[0]].order
    n_coeffs = benchmark.X_train.shape[1]
    setup = _MATH_SETUP.format(
        degree=degree,
        n_coeffs=n_coeffs,
        classes=" vs ".join(benchmark.class_names),
        order_tables=_order_tables(benchmark.class_names),
    )
    contract = _OUTPUT_CONTRACT.format(
        n_coeffs=n_coeffs, label_map=_label_map(benchmark.class_names)
    )
    if protocol == "derivation":
        task_text = (
            "Task: from the mathematics alone (no data is provided), "
            "derive a criterion on the zeta coefficients that determines "
            "the Galois group, and implement it.\n"
        )
        return f"{setup}\n{task_text}\n{contract}"
    if protocol == "rule_induction":
        task_text = (
            "Task: infer the rule that determines the Galois group from "
            "the zeta coefficients. Use the data below and the "
            "mathematical facts above. State the rule and implement it.\n"
            f"\nExample rows:\n{data_excerpt(benchmark, n_examples, seed)}\n"
            f"\nEmpirical value distributions of prime-power coefficients "
            f"(training split):\n{prime_power_summary(benchmark)}\n"
        )
        return f"{setup}\n{task_text}\n{contract}"
    raise ValueError(f"unknown protocol {protocol!r}")


def build_classification_prompt(
    benchmark: GaloisBenchmark,
    n_shots: int = 8,
    n_queries: int = 40,
    seed: int = 0,
) -> tuple[str, np.ndarray]:
    """Few-shot direct classification prompt. Returns (prompt, indices
    of the queried test rows)."""
    degree = GROUPS[benchmark.class_names[0]].order
    n_coeffs = benchmark.X_train.shape[1]
    rng = np.random.default_rng(seed)
    setup = _MATH_SETUP.format(
        degree=degree,
        n_coeffs=n_coeffs,
        classes=" vs ".join(benchmark.class_names),
        order_tables=_order_tables(benchmark.class_names),
    )
    shots = data_excerpt(benchmark, n_examples=n_shots, seed=seed)
    queries = rng.choice(len(benchmark.y_test), size=n_queries, replace=False)
    query_lines = []
    for qi, i in enumerate(queries):
        row = ", ".join(str(int(v)) for v in benchmark.X_test[i])
        query_lines.append(f"  query {qi}: [{row}]")
    prompt = (
        f"{setup}\nLabeled examples:\n{shots}\n\n"
        f"Classify each query row ({_label_map(benchmark.class_names)}).\n"
        + "\n".join(query_lines)
        + '\n\nRespond with ONLY a JSON object: {"labels": [<int>, ...]} '
        f"with exactly {n_queries} integers in query order.\n"
    )
    return prompt, queries


# -- response parsing and sandboxed execution ------------------------------

def _balanced_objects(text: str) -> list[str]:
    """All top-level balanced {...} spans in `text`, tracking string
    literals so braces inside quoted strings do not confuse the scan."""
    spans, stack_depth, start = [], 0, -1
    in_string, escaped = False, False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if stack_depth == 0:
                start = i
            stack_depth += 1
        elif ch == "}" and stack_depth > 0:
            stack_depth -= 1
            if stack_depth == 0:
                spans.append(text[start : i + 1])
    return spans


def extract_json(text: str) -> dict:
    """Extract the answer JSON object from a model response. Tolerates
    code fences, surrounding prose, and stray braces in earlier text
    (e.g. tool outputs quoted in an agent transcript): candidates are
    balanced brace spans, tried from the end of the response, where the
    final answer lives."""
    candidates = []
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    candidates.extend(reversed(_balanced_objects(text)))
    for cand in candidates:
        try:
            parsed = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("no parseable JSON object in response")


_DRIVER = """\
import json, sys
payload = json.loads(sys.stdin.read())
namespace = {}
exec(payload["code"], namespace)
predict = namespace["predict"]
print(json.dumps([int(predict(row)) for row in payload["rows"]]))
"""


def run_predict_code(
    code: str, X: np.ndarray, timeout: int = CODE_TIMEOUT_SECONDS
) -> np.ndarray:
    """Execute an untrusted candidate rule in an isolated subprocess
    (python -I: no site-packages, no user paths) with a timeout, and
    return its predictions on the rows of X."""
    payload = json.dumps({"code": code, "rows": X.tolist()})
    proc = subprocess.run(
        [sys.executable, "-I", "-c", _DRIVER],
        input=payload,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"candidate code failed: {proc.stderr[-500:]}")
    return np.array(json.loads(proc.stdout), dtype=np.int64)


# -- evaluation ------------------------------------------------------------

def evaluate_model(
    model,
    benchmark: GaloisBenchmark,
    protocol: str = "rule_induction",
    model_name: str = "model",
    seed: int = 0,
) -> EvalResult:
    """Run one model on one benchmark task under a protocol and score it
    against the ground truth. `model` is any callable prompt -> text."""
    base = dict(
        task=benchmark.task,
        protocol=protocol,
        model_name=model_name,
        bayes_accuracy=benchmark.bayes_accuracy,
        n_test=len(benchmark.y_test),
    )
    raw = None

    def _ask_and_parse(prompt: str) -> dict:
        """One retry on unparseable output: re-prompt with a reminder
        (typical causes: truncation, prose-wrapped or malformed JSON)."""
        nonlocal raw
        raw = model(prompt)
        try:
            return extract_json(raw)
        except ValueError:
            raw = model(
                prompt
                + "\n\nREMINDER: your entire response must be a single "
                "valid JSON object, with no surrounding text."
            )
            return extract_json(raw)

    try:
        if protocol in ("rule_induction", "derivation"):
            prompt = build_prompt(benchmark, protocol, seed=seed)
            parsed = _ask_and_parse(prompt)
            code, conjecture = parsed["python"], parsed.get("conjecture")
            preds = run_predict_code(code, benchmark.X_test)
            acc = float(np.mean(preds == benchmark.y_test))
        elif protocol == "direct_classification":
            prompt, queries = build_classification_prompt(
                benchmark, seed=seed
            )
            parsed = _ask_and_parse(prompt)
            labels = np.array(parsed["labels"], dtype=np.int64)
            truth = benchmark.y_test[queries]
            if len(labels) != len(truth):
                raise ValueError(
                    f"expected {len(truth)} labels, got {len(labels)}"
                )
            acc = float(np.mean(labels == truth))
            conjecture = None
            base["n_test"] = len(truth)
        else:
            raise ValueError(f"unknown protocol {protocol!r}")
    except Exception as exc:  # malformed output, crash, timeout, ...
        return EvalResult(
            **base,
            valid_response=False,
            accuracy=None,
            regret=None,
            conjecture=None,
            error=f"{type(exc).__name__}: {exc}",
            raw_response=raw[:4000] if raw else None,
        )
    return EvalResult(
        **base,
        valid_response=True,
        accuracy=acc,
        regret=benchmark.bayes_accuracy - acc,
        conjecture=conjecture,
        raw_response=raw[:4000] if raw else None,
    )


# -- offline baselines (validate the harness without any API) --------------

_ORACLE_CYCLIC_TEMPLATE = """\
def predict(zeta):
    # Gal is cyclic of order l^2 (or l^m) iff some prime is inert,
    # equivalently a_(p^l) = 0 for some p (Cor 3.3 pattern, l = {l}).
    def is_prime(m):
        return m > 1 and all(m % d for d in range(2, int(m**0.5) + 1))
    n = len(zeta)
    p = 2
    while p ** {l} <= n:
        if is_prime(p) and zeta[p ** {l} - 1] == 0:
            return 0
        p += 1
    return 1
"""

#: task -> the exponent l such that "a_(p^l) = 0 for some p" certifies
#: the cyclic group. Covers the degree-l^2 family.
_ORACLE_EXPONENT = {"quartic": 2, "nonic": 3, "degree25": 5}


class OracleModel:
    """Simulates a model that answers with the proven theorem. Should
    score at (or extremely near) the Bayes ceiling - the harness's
    positive control."""

    def __init__(self, task: str):
        if task not in _ORACLE_EXPONENT:
            raise ValueError(f"no oracle rule wired for task {task!r}")
        self.task = task

    def __call__(self, prompt: str) -> str:
        l = _ORACLE_EXPONENT[self.task]
        return json.dumps({
            "conjecture": f"The group is cyclic iff a_(p^{l}) = 0 for "
                          f"some prime p (inert primes exist only in the "
                          f"cyclic case).",
            "python": _ORACLE_CYCLIC_TEMPLATE.format(l=l),
        })


class MajorityModel:
    """Answers with a constant-label rule - the harness's floor control."""

    def __init__(self, label: int = 0):
        self.label = label

    def __call__(self, prompt: str) -> str:
        return json.dumps({
            "conjecture": "no rule found; predicting the majority class",
            "python": f"def predict(zeta):\n    return {self.label}",
        })


# -- Bedrock adapter (used once credentials are configured) ----------------

class BedrockModel:
    """Amazon Bedrock chat model as a prompt -> text callable.

    Requires AWS credentials with bedrock:InvokeModel (e.g. via
    `aws configure` or environment variables). Uses the Converse API,
    so any text model id works (Claude, Nova, Llama, ...).
    """

    def __init__(
        self,
        model_id: str,
        region: str = "us-east-1",
        max_tokens: int = 16000,
        temperature: float = 0.0,
    ):
        import boto3  # deferred: only needed when actually used

        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature

    def __call__(self, prompt: str) -> str:
        response = self.client.converse(
            modelId=self.model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={
                "maxTokens": self.max_tokens,
                "temperature": self.temperature,
            },
        )
        return response["output"]["message"]["content"][0]["text"]
