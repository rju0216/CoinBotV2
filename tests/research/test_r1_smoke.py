"""R1 스모크 사전등록 판정규칙 박제 (규칙 17).

실 데이터 R1 실행 자체는 수분 소요라 회귀에 부적합(`python -m ...`로 1회 실행). 여기선
**판정규칙 로직**(``classify_gate1``/``_eval_flags``)이 사전등록 규칙대로 동작함을 못박아,
규칙이 조용히 바뀌면 회귀가 잡게 한다. 규칙 = 3조건(ll<Prior ∧ mcc>0 ∧ 과반 일관성),
비대칭(엄격PASS/관대DISCARD/HOLD완충).
"""

from __future__ import annotations

import numpy as np

from src.research.experiments.r1_smoke import (
    PREREGISTERED_RULE,
    ModelEval,
    _eval_flags,
    classify_gate1,
)

PRIOR_LL = float(np.log(3))   # 기준 Prior log_loss (설명용; 실행 시엔 실측)


def _ev(name, ll, mcc, cons, ba=0.34):
    return _eval_flags(ModelEval(name, ll, mcc, ba, cons), PRIOR_LL)


def test_discard_when_no_model_beats_prior():
    evals = {"tree": _ev("tree", 1.10, 0.0, 0.3), "mlp": _ev("mlp", 1.11, -0.01, 0.2)}
    verdict, _ = classify_gate1(evals, PRIOR_LL)
    assert verdict == "DISCARD"


def test_pass_when_mlp_strong():
    # 3조건 전부 (ll<Prior, mcc>0, 과반). BA 낮아도 통과해야(게이트 아님).
    evals = {"tree": _ev("tree", 1.10, 0.0, 0.4), "mlp": _ev("mlp", 1.05, 0.05, 0.7, ba=0.30)}
    verdict, _ = classify_gate1(evals, PRIOR_LL)
    assert verdict == "PASS"
    assert evals["mlp"].strong and evals["mlp"].weak


def test_pass_via_tree():
    evals = {"tree": _ev("tree", 1.04, 0.06, 0.65), "mlp": _ev("mlp", 1.11, 0.0, 0.3)}
    verdict, reason = classify_gate1(evals, PRIOR_LL)
    assert verdict == "PASS"
    assert "tree" in reason


def test_hold_when_weak_but_not_strong():
    # ll<Prior ∧ mcc>0 (WEAK) 이나 일관성 과반 미달 → HOLD (즉시폐기 금지, A-1)
    evals = {"tree": _ev("tree", 1.10, 0.0, 0.4), "mlp": _ev("mlp", 1.08, 0.02, 0.45)}
    verdict, _ = classify_gate1(evals, PRIOR_LL)
    assert verdict == "HOLD"
    assert evals["mlp"].weak and not evals["mlp"].strong


def test_exactly_half_is_not_majority():
    # 정확히 50% 일관성 → 과반 아님 → STRONG 아님 → HOLD (경계 엄격)
    evals = {"tree": _ev("tree", 1.10, 0.0, 0.4), "mlp": _ev("mlp", 1.08, 0.02, 0.50)}
    verdict, _ = classify_gate1(evals, PRIOR_LL)
    assert verdict == "HOLD"


def test_ba_not_in_gate():
    # BA 가 낮아도(0.20) 3조건 충족 시 STRONG — BA 는 보고 지표일 뿐 게이트 아님
    ev = _ev("mlp", 1.05, 0.05, 0.7, ba=0.20)
    assert ev.strong


def test_prior_uniform_never_strong():
    # 기준선은 상수예측(mcc=0) → 절대 STRONG/WEAK 아님
    prior = _ev("prior", 1.05, 0.0, 1.0)     # ll 자기 자신 대비이나 mcc=0
    uniform = _ev("uniform", 1.0986, 0.0, 0.0)
    assert not prior.weak and not prior.strong
    assert not uniform.weak and not uniform.strong


def test_rule_text_states_three_conditions():
    # 사전등록 규칙 텍스트가 3조건·비대칭을 명시(문서-코드 정합, 규칙 13)
    assert "ll_median < Prior_ll_median" in PREREGISTERED_RULE
    assert "mcc_median > 0" in PREREGISTERED_RULE
    assert "과반" in PREREGISTERED_RULE
    assert "BA" in PREREGISTERED_RULE  # 보고 지표 명시
