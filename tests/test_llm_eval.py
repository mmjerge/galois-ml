import json

import numpy as np
import pytest

from galois_ml.benchmark import load_benchmark
from galois_ml.llm_eval import (
    MajorityModel,
    OracleModel,
    build_classification_prompt,
    build_prompt,
    evaluate_model,
    extract_json,
    run_predict_code,
)


@pytest.fixture(scope="module")
def quartic_bench():
    return load_benchmark(
        "quartic", n_train=200, n_test=100, n_coeffs=200, bayes_samples=20000
    )


def test_extract_json_variants():
    obj = {"conjecture": "x", "python": "def predict(zeta):\n    return 0"}
    raw = json.dumps(obj)
    assert extract_json(raw) == obj
    assert extract_json(f"Here is my answer:\n```json\n{raw}\n```\nDone.") == obj
    assert extract_json(f"prose before {raw}") == obj
    with pytest.raises(ValueError):
        extract_json("no json here at all")


def test_run_predict_code_executes_and_isolates():
    X = np.array([[1.0, 0.0], [1.0, 5.0]])
    preds = run_predict_code(
        "def predict(zeta):\n    return 1 if zeta[1] > 0 else 0", X
    )
    assert preds.tolist() == [0, 1]
    # crashing code surfaces as an error, not a hang or bogus score
    with pytest.raises(RuntimeError):
        run_predict_code("def predict(zeta):\n    raise ValueError('boom')", X)


def test_prompts_contain_the_contract(quartic_bench):
    for protocol in ("rule_induction", "derivation"):
        prompt = build_prompt(quartic_bench, protocol)
        assert "predict(zeta)" in prompt
        assert "C4" in prompt and "C2^2" in prompt
        assert "a_(p^k)" in prompt  # the Euler factor fact sheet
    # derivation must not leak data
    assert "label 0" not in build_prompt(quartic_bench, "derivation")
    assert "label 0" in build_prompt(quartic_bench, "rule_induction")


def test_classification_prompt_query_alignment(quartic_bench):
    prompt, queries = build_classification_prompt(
        quartic_bench, n_shots=2, n_queries=5
    )
    assert len(queries) == 5
    assert all(f"query {i}: [" in prompt for i in range(5))
    assert "query 5: [" not in prompt


def test_oracle_hits_ceiling_and_majority_hits_floor(quartic_bench):
    oracle = evaluate_model(
        OracleModel("quartic"), quartic_bench, "rule_induction", "oracle"
    )
    assert oracle.valid_response
    # theorem rule should be within noise of the Bayes ceiling
    assert oracle.accuracy >= quartic_bench.bayes_accuracy - 0.03

    floor = evaluate_model(
        MajorityModel(0), quartic_bench, "rule_induction", "majority"
    )
    assert floor.valid_response
    assert abs(floor.accuracy - np.mean(quartic_bench.y_test == 0)) < 1e-9
    assert floor.regret > oracle.regret


def test_malformed_model_yields_invalid_not_crash(quartic_bench):
    res = evaluate_model(
        lambda prompt: "I refuse to answer with JSON.",
        quartic_bench,
        "rule_induction",
        "broken",
    )
    assert not res.valid_response
    assert res.accuracy is None and res.error is not None


def test_held_out_task_flows_through_harness():
    bench = load_benchmark(
        "degree25", n_train=100, n_test=60, n_coeffs=300, bayes_samples=20000
    )
    res = evaluate_model(
        OracleModel("degree25"), bench, "rule_induction", "oracle"
    )
    assert res.valid_response
    assert res.accuracy >= bench.bayes_accuracy - 0.05
