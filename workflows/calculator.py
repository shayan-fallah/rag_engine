"""The agent's calculator tool: exact arithmetic for expressions a model writes.

Models write digits that look right, which is not the same as computing them. With
the calculator an agent sends the expressions it needs and copies exact results.

Expressions are parsed with Python's ast and only arithmetic is evaluated (no names,
attributes or other calls), so a model can't run code with it. Numbers are exact
fractions: 1/3 + 1/6 is 1/2, 0.1 + 0.2 is 0.3. Only sqrt of a non-square, fractional
powers and pi give approximations. Sizes are limited, so a model can't make it compute
for minutes.
"""

import ast
import math
import re
from fractions import Fraction

DESCRIPTION = (
    "Exact calculator. Use it for every calculation with numbers instead of calculating in your head, and copy its "
    "results exactly. Send all the calculations you need in one call, one per expression, e.g. every step of a long "
    "multiplication and the final result. A step that needs an earlier result can repeat its expression: "
    '["345 * 7", "345 * 20", "345 * 7 + 345 * 20"].\n'
    "Write: + - * / and ( ), ** for powers, decimals with a dot (12.75), no thousands separators, a fraction as 3/4, "
    "a mixed number as (5 + 2/3), a percent as 15/100 * 3480. Functions: sqrt(x), a mod b (the remainder), a // b "
    "(whole-number division), gcd(a, b), lcm(a, b), round(x, digits) (halves round up), floor(x), ceil(x), abs(x), "
    "min(...), max(...), factorial(n), pi. Fractions come back exact and simplified, also as a mixed number and as a "
    "decimal."
)

MAX_LENGTH = 300  # characters in one expression
MAX_DIGITS = 1000  # digits of any number along the way
MAX_EXPRESSIONS = 30  # per call
SHOW_DIGITS = 60  # longer exact numbers are shown rounded

# Persian and Arabic digits, and symbols models copy from the question.
_NORMALIZE = str.maketrans(
    {
        **{c: str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")},
        **{c: str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")},
        "×": "*",
        "÷": "/",
        "−": "-",
        "–": "-",
        "٫": ".",
        "،": ",",
        "^": "**",
    }
)


class CalcError(ValueError):
    pass


def _whole(x, name: str) -> int:
    if isinstance(x, Fraction) and x.denominator == 1:
        return x.numerator
    raise CalcError(f"{name} needs whole numbers")


def _sqrt(x):
    if x < 0:
        raise CalcError("sqrt of a negative number")
    if isinstance(x, Fraction):
        n, d = math.isqrt(x.numerator), math.isqrt(x.denominator)
        if n * n == x.numerator and d * d == x.denominator:
            return Fraction(n, d)
    return math.sqrt(x)


def _mod(a, b):
    if b == 0:
        raise CalcError("mod by zero")
    return a % b


def _round(x, digits=Fraction(0)):
    """Halves round away from zero, as taught in school (2.5 -> 3), not to the even number."""
    d = _whole(digits, "round's digits")
    if not -20 <= d <= 50:
        raise CalcError("round's digits must be between -20 and 50")
    value = Fraction(x)
    scale = Fraction(10) ** d
    rounded = Fraction(math.floor(abs(value) * scale + Fraction(1, 2))) / scale
    return rounded if value >= 0 else -rounded


def _factorial(n):
    n = _whole(n, "factorial")
    if not 0 <= n <= 300:
        raise CalcError("factorial works from 0 to 300")
    return Fraction(math.factorial(n))


FUNCTIONS = {
    "sqrt": _sqrt,
    "mod": _mod,
    "gcd": lambda *xs: Fraction(math.gcd(*(_whole(x, "gcd") for x in xs))),
    "lcm": lambda *xs: Fraction(math.lcm(*(_whole(x, "lcm") for x in xs))),
    "round": _round,
    "floor": lambda x: Fraction(math.floor(x)),
    "ceil": lambda x: Fraction(math.ceil(x)),
    "abs": abs,
    "min": min,
    "max": max,
    "factorial": _factorial,
}
CONSTANTS = {"pi": math.pi}


def _check_size(x):
    if isinstance(x, float):
        if not math.isfinite(x):
            raise CalcError("the result is too large")
    elif max(len(str(abs(x.numerator))), len(str(x.denominator))) > MAX_DIGITS:
        raise CalcError(f"a number got longer than {MAX_DIGITS} digits")
    return x


def _power(base, exp):
    if isinstance(base, Fraction) and isinstance(exp, Fraction) and exp.denominator == 1:
        e = exp.numerator
        if base == 0 and e < 0:
            raise CalcError("division by zero")
        size = max(abs(base.numerator), base.denominator)
        if size > 1 and abs(e) * math.log10(size) > MAX_DIGITS:
            raise CalcError(f"the result would have more than {MAX_DIGITS} digits")
        return base**e
    if base < 0:
        raise CalcError("a negative number to a fractional power")
    try:
        return float(base) ** float(exp)
    except OverflowError:
        raise CalcError("the result is too large") from None


def _number(node: ast.Constant, text: str) -> Fraction:
    source = (ast.get_source_segment(text, node) or str(node.value)).replace("_", "")
    exponent = source.lower().partition("e")[2] if not source.lower().startswith("0x") else ""
    if exponent.lstrip("+-").isdigit() and abs(int(exponent)) > MAX_DIGITS:
        raise CalcError(f"a number got longer than {MAX_DIGITS} digits")
    try:
        value = Fraction(source)  # the digits as written: 0.1 is exactly 1/10
    except (ValueError, ZeroDivisionError):
        value = Fraction(str(node.value))  # e.g. 0x10
    return _check_size(value)


def _eval(node, text: str):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return _number(node, text)
    if isinstance(node, ast.Name):
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]
        raise CalcError(f"unknown name {node.id!r}; write numbers only (no variables)")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
        value = _eval(node.operand, text)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        a, b = _eval(node.left, text), _eval(node.right, text)
        op = node.op
        if isinstance(op, ast.Mod):
            return _mod(a, b)
        if isinstance(op, ast.Add):
            return _check_size(a + b)
        if isinstance(op, ast.Sub):
            return _check_size(a - b)
        if isinstance(op, ast.Mult):
            return _check_size(a * b)
        if isinstance(op, ast.Div | ast.FloorDiv):
            if b == 0:
                raise CalcError("division by zero")
            return _check_size(a / b if isinstance(op, ast.Div) else Fraction(math.floor(a / b)))
        if isinstance(op, ast.Pow):
            return _check_size(_power(a, b))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        func = FUNCTIONS.get(node.func.id)
        if func is None:
            raise CalcError(f"unknown function {node.func.id!r}")
        args = [_eval(arg, text) for arg in node.args]
        try:
            return _check_size(func(*args))
        except TypeError:
            raise CalcError(f"wrong number of values for {node.func.id}()") from None
    raise CalcError("only numbers, + - * / // ** ( ) and the listed functions can be used")


def evaluate(expression: str):
    """The exact value of an arithmetic expression: a Fraction, or a float for sqrt/pi/fractional powers."""
    text = expression.translate(_NORMALIZE).strip().rstrip("=").strip()
    if not text:
        raise CalcError("empty expression")
    if len(text) > MAX_LENGTH:
        raise CalcError(f"expressions can be at most {MAX_LENGTH} characters")
    if "%" in text or "٪" in text:
        raise CalcError("% is not supported: write a mod b for a remainder, or 15/100 * x for a percent")
    text = re.sub(r"\bmod\b(?!\s*\()", "%", text)  # a mod b; % only ever comes from here
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, RecursionError, MemoryError):
        raise CalcError("this is not an arithmetic expression") from None
    return _eval(tree.body, text)


def _decimal(x: Fraction) -> str | None:
    """The exact decimal of a fraction whose denominator has only the factors 2 and 5 (short ones), else None."""
    d, twos, fives = x.denominator, 0, 0
    while d % 2 == 0:
        d, twos = d // 2, twos + 1
    while d % 5 == 0:
        d, fives = d // 5, fives + 1
    places = max(twos, fives)
    if d != 1 or places > 20:
        return None
    whole, frac = divmod(abs(x.numerator) * 10**places // x.denominator, 10**places)
    return f"{'-' if x < 0 else ''}{whole}.{str(frac).zfill(places)}"


def _approx(x) -> str:
    text = f"{float(x):.10g}"
    return text if "e" in text or "." not in text else text.rstrip("0").rstrip(".")


def format_value(x) -> str:
    """How a result is written for the model: '= 85/4 = 21 1/4 = 21.25', '= 68/77 ≈ 0.8831168831'."""
    if isinstance(x, float):
        return f"≈ {_approx(x)}"
    longest = max(len(str(abs(x.numerator))), len(str(x.denominator)))
    if longest > SHOW_DIGITS:
        return f"≈ {_approx(x)} (exact value too long to show)"
    if x.denominator == 1:
        return f"= {x.numerator}"
    parts = [f"{x.numerator}/{x.denominator}"]
    if abs(x) > 1:
        whole = math.trunc(x)
        rest = abs(x - whole)
        parts.append(f"{whole} {rest.numerator}/{rest.denominator}")
    exact = _decimal(x)
    return "= " + " = ".join(parts) + (f" = {exact}" if exact else f" ≈ {_approx(x)}")


def calculate(expressions: list[str]) -> str:
    """One line per expression: 'expression = result', or 'expression → error: ...' (the others still run)."""
    if not isinstance(expressions, list) or not expressions:
        raise ValueError("expressions must be a non-empty list of strings")
    if len(expressions) > MAX_EXPRESSIONS:
        raise ValueError(f"send at most {MAX_EXPRESSIONS} expressions per call")
    lines = []
    for expression in expressions:
        expression = str(expression).strip().rstrip("=").strip()
        try:
            lines.append(f"{expression} {format_value(evaluate(expression))}")
        except (ArithmeticError, ValueError, TypeError, RecursionError) as exc:
            lines.append(f"{expression} → error: {exc}")
    return "\n".join(lines)
