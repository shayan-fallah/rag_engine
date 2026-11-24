import pytest

from tests.fakes import ScriptedLLM
from workflows.calculator import calculate, evaluate
from workflows.dsl import validate_dsl
from workflows.engine import run_workflow
from workflows.models import Workflow


@pytest.mark.parametrize(
    "expression, result",
    [
        ("4738 * 296", "= 1402448"),
        ("7865 × 4093", "= 32191445"),
        ("۳۴۵ × ۲۷", "= 9315"),  # Persian digits
        ("98765 // 37", "= 2669"),
        ("mod(98765, 37)", "= 12"),
        ("98765 mod 37", "= 12"),
        ("3/7 + 5/11", "= 68/77 ≈ 0.8831168831"),
        ("(5 + 2/3) * (3 + 3/4)", "= 85/4 = 21 1/4 = 21.25"),
        ("12.75 * 3.6", "= 459/10 = 45 9/10 = 45.9"),
        ("0.1 + 0.2", "= 3/10 = 0.3"),  # exact, not 0.30000000000000004
        ("15/100 * 3480", "= 522"),
        ("2^10", "= 1024"),
        ("-85/4", "= -85/4 = -21 1/4 = -21.25"),
        ("sqrt(1764)", "= 42"),
        ("sqrt(2)", "≈ 1.414213562"),
        ("round(2.5)", "= 3"),  # halves round up, as in school
        ("round(-2.5)", "= -3"),
        ("round(3.14159, 2)", "= 157/50 = 3 7/50 = 3.14"),
        ("gcd(12, 18) + lcm(4, 6)", "= 18"),
        ("factorial(10)", "= 3628800"),
        ("pi * 3**2", "≈ 28.27433388"),
        ("1/2**80", "= 1/1208925819614629174706176 ≈ 8.271806126e-25"),
    ],
)
def test_calculator_results(expression, result):
    assert calculate([expression]) == f"{expression} {result}"


@pytest.mark.parametrize(
    "expression, error",
    [
        ("1/0", "division by zero"),
        ("15% * 3480", "write a mod b for a remainder, or 15/100 * x for a percent"),
        ("7 mod 0", "mod by zero"),
        ("x + 5", "unknown name 'x'"),
        ("__import__('os')", "unknown function '__import__'"),
        ("__import__('os').system('ls')", "only numbers"),
        ("(1).__class__", "only numbers"),
        ("'a' * 3", "only numbers"),
        ("True + 1", "only numbers"),
        ("1,000", "only numbers"),
        ("9**9**9", "more than 1000 digits"),
        ("2**2**2**2**2", "more than 1000 digits"),
        ("1e999999999", "longer than 1000 digits"),
        ("10**999 * 10", "longer than 1000 digits"),
        ("factorial(1000)", "factorial works from 0 to 300"),
        ("(-8) ** (1/3)", "a negative number to a fractional power"),
        ("min()", "wrong number of values"),
        ("1 +", "not an arithmetic expression"),
        ("1" * 301, "at most 300 characters"),
    ],
)
def test_calculator_refuses(expression, error):
    with pytest.raises(ValueError, match=error.replace("(", r"\(").replace(")", r"\)").replace("*", r"\*")):
        evaluate(expression)


def test_one_bad_expression_doesnt_stop_the_others():
    assert calculate(["1/0", "2 + 2 ="]) == "1/0 → error: division by zero\n2 + 2 = 4"
    with pytest.raises(ValueError, match="at most 30"):
        calculate(["1"] * 31)
    with pytest.raises(ValueError, match="non-empty list"):
        calculate("1 + 1")


def tutor(**params):
    return {
        "nodes": {
            "begin": {"type": "begin", "next": "tutor"},
            "tutor": {"type": "agent", "params": {"calculator": True, **params}, "next": "reply"},
            "reply": {"type": "message", "params": {"text": "{tutor@content}"}},
        }
    }


def test_agent_with_only_a_calculator(kb, use_llm):
    llm = use_llm(
        ScriptedLLM(
            tool_plan=lambda s, m: [("calculate", {"expressions": ["345 * 7", "345 * 20", "2415 + 6900"]})],
            reply=lambda s, m: "۳۴۵ × ۲۷ = ۹۳۱۵",
        )
    )
    wf = Workflow.objects.create(name="tutor", dsl=tutor())
    validate_dsl(wf.dsl)
    events = list(run_workflow(wf, "345 × 27?", user_id="u1"))
    calls = [e for e in events if e["event"] == "tool_call"]
    assert len(calls) == 1 and calls[0]["tool"] == "calculate" and calls[0]["kind"] == "calculator"
    assert calls[0]["results"] == 3 and "result" not in calls[0]  # the result only in debug runs
    assert llm.calls[-1]["messages"][-1]["content"] == "345 * 7 = 2415\n345 * 20 = 6900\n2415 + 6900 = 9315"
    assert events[-1]["answer"] == "۳۴۵ × ۲۷ = ۹۳۱۵"


def test_calculator_settings_are_checked(kb):
    with pytest.raises(ValueError, match="add a search tool or turn on the calculator"):
        validate_dsl(tutor(calculator=False))
    tool = {"name": "calculate", "description": "d", "kb_ids": [kb.id]}
    with pytest.raises(ValueError, match="can't be called 'calculate'"):
        validate_dsl(tutor(tools=[tool]))
    validate_dsl(tutor(tools=[{**tool, "name": "search"}], calculator_description="Use it for sums."))
