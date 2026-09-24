from __future__ import annotations

"""
Adapted from the official omni-math-rule evaluator:
https://github.com/KbsdJames/omni-math-rule
"""

import multiprocessing
import re
from math import isclose
from typing import Any, Dict, List, Optional, Union

try:
    import regex as regex_mod
except ImportError:  # pragma: no cover - fallback is acceptable
    regex_mod = re

try:  # pragma: no cover - import availability is environment-dependent
    from sympy import N, simplify
    from sympy.parsing.latex import parse_latex
    from sympy.parsing.sympy_parser import parse_expr
except ImportError:  # pragma: no cover
    N = simplify = parse_latex = parse_expr = None

try:  # pragma: no cover - import availability is environment-dependent
    from latex2sympy2 import latex2sympy
except ImportError:  # pragma: no cover
    latex2sympy = None

try:  # pragma: no cover - optional helper
    from word2number import w2n
except ImportError:  # pragma: no cover
    w2n = None


UNIT_TEXTS = [
    "east", "degree", "mph", "kmph", "ft", "sq m", "deg", "mile", "lb", "tile",
    "per", "dm", "lt", "gain", "inches", "percent", "gal", "acre", "sq", "profit",
    "yr", "feet", "am", "pm", "hr", "square", "are", "rupee", "rounds", "cc",
    "number", "day", "hour", "minute", "min", "second", "man", "woman", "sec",
    "unit", "rs", "kg", "g", "month", "km", "m", "cm", "mm", "liter", "loss",
    "yard", "year", "increase", "decrease", "meter", "metre", "inch", "dollar",
    "dollars", "cents", "hours", "minutes", "seconds", "days", "weeks", "months",
    "years", "meters", "kilometers", "centimeters", "millimeters", "liters",
]
UNIT_TEXTS.extend([f"{text}s" for text in UNIT_TEXTS])
_OMNI_RULE_CACHE: Dict[str, "OmniRuleEvaluator"] = {}


def _ensure_dependencies() -> None:
    missing = []
    if N is None or simplify is None or parse_latex is None or parse_expr is None:
        missing.append("sympy")
    if latex2sympy is None:
        missing.append("latex2sympy2")
    if missing:
        deps = " ".join(sorted(set(missing + ["antlr4-python3-runtime", "word2number", "regex"])))
        raise RuntimeError(
            "Omni-rule evaluation requires additional packages. "
            f"Install at least: {deps}"
        )


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    post_substr = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}{" + b + "}" + post_substr
                else:
                    post_substr = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}" + b + post_substr
    return new_str


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        if string == f"{a}/{b}":
            return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except Exception:
        pass
    return string


def _fix_sqrt(string: str) -> str:
    return re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)


def _convert_word_number(text: str) -> str:
    if w2n is None:
        return text
    try:
        return str(w2n.word_to_num(text))
    except Exception:
        return text


def _strip_string(string: Any, skip_unit: bool = False) -> str:
    string = str(string).strip()
    string = string.replace("\n", "")
    string = string.rstrip(".")
    string = string.replace("\\!", "")
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\neq", "\\ne").replace("\\leq", "\\le").replace("\\geq", "\\ge")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("\\{", "{").replace("\\}", "}")

    text_tail_removed = re.sub(r"\\text{.*?}$", "", string).strip()
    if text_tail_removed:
        string = text_tail_removed

    if not skip_unit:
        for _ in range(2):
            for unit_text in UNIT_TEXTS:
                string = re.sub(r"(^|\W)" + re.escape(unit_text) + r"($|\W)", r"\1\2", string)

    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "").replace("$", "")
    string = string.replace("\\(", "").replace("\\)", "")
    string = _convert_word_number(string)
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    for key in ["x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to"]:
        string = string.replace(key, "")
    string = string.replace("\\emptyset", r"{}")
    string = string.replace("(-\\infty,\\infty)", "\\mathbb{R}")
    string = string.replace("\\%", "").replace("%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")
    string = string.replace("and", "").replace("\\mathbf", "")
    string = re.sub(r"\\mbox{.*?}", "", string)

    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    string = _fix_a_slash_b(string)
    return string


def _extract_answer(pred_str: str, data_name: str = "omni-math", use_last_number: bool = True) -> str:
    pred_str = (pred_str or "").replace("\u043a\u0438", "")

    if "final answer is $" in pred_str and "$. I hope" in pred_str:
        pred = pred_str.split("final answer is $", 1)[1].split("$. I hope", 1)[0].strip()
    elif "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if not ans:
            pred = ""
        elif ans[0] == "{":
            stack = 1
            buf = []
            for char in ans[1:]:
                if char == "{":
                    stack += 1
                    buf.append(char)
                elif char == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    buf.append(char)
                else:
                    buf.append(char)
            pred = "".join(buf)
        else:
            pred = ans.split("$")[0].strip()
    elif "he answer is" in pred_str:
        pred = pred_str.split("he answer is")[-1].strip()
    elif "final answer is" in pred_str:
        pred = pred_str.split("final answer is")[-1].strip()
    elif use_last_number:
        pattern = r"-?\d*\.?\d+"
        matches = re.findall(pattern, pred_str.replace(",", ""))
        pred = matches[-1] if matches else ""
    else:
        pred = ""

    pred = re.sub(r"\n\s*", "", pred)
    if pred.startswith(":"):
        pred = pred[1:]
    pred = pred.rstrip("./")
    return _strip_string(pred, skip_unit=data_name in ["carp_en", "minerva_math"])


def _parse_ground_truth(answer: str, data_name: str = "omni-math") -> str:
    gt_cot = str(answer)
    if data_name == "omni-math" and "boxed" not in gt_cot:
        gt_cot = "\\boxed{" + gt_cot + "}"
    return _extract_answer(gt_cot, data_name)


def _choice_answer_clean(pred: str) -> str:
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    pred = tmp[-1] if tmp else pred.strip().strip(".")
    return pred.rstrip(".").rstrip("/")


def _parse_digits(num: Any) -> Optional[float]:
    num = regex_mod.sub(",", "", str(num))
    try:
        return float(num)
    except Exception:
        if str(num).endswith("%"):
            stripped = str(num)[:-1].rstrip("\\")
            try:
                return float(stripped) / 100
            except Exception:
                return None
    return None


def _is_digit(num: Any) -> bool:
    return _parse_digits(num) is not None


def _str_to_pmatrix(input_str: str) -> str:
    matrix_str = re.findall(r"\{.*,.*\}", input_str.strip())
    pmatrix_list = []
    for item in matrix_str:
        item = item.strip("{}")
        pmatrix_list.append(r"\begin{pmatrix}" + item.replace(",", "\\") + r"\end{pmatrix}")
    return ", ".join(pmatrix_list)


def _numeric_equal(prediction: float, reference: float) -> bool:
    return isclose(reference, prediction, rel_tol=1e-4)


def _symbolic_equal(a: str, b: str) -> bool:
    def _parse(s: str):
        for parser in [parse_latex, parse_expr, latex2sympy]:
            try:
                return parser(s.replace("\\\\", "\\"))
            except Exception:
                try:
                    return parser(s)
                except Exception:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    try:
        if str(a) == str(b) or a == b:
            return True
    except Exception:
        pass

    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except Exception:
        pass

    try:
        if abs(a.lhs - a.rhs).equals(abs(b.lhs - b.rhs)):
            return True
    except Exception:
        pass

    try:
        if _numeric_equal(float(N(a)), float(N(b))):
            return True
    except Exception:
        pass

    try:
        if a.shape == b.shape:
            if a.applyfunc(lambda x: round(x, 3)).equals(b.applyfunc(lambda x: round(x, 3))):
                return True
    except Exception:
        pass

    return False


def _symbolic_equal_process(a: str, b: str, output_queue: multiprocessing.Queue) -> None:
    output_queue.put(_symbolic_equal(a, b))


def _call_with_timeout(func, *args, timeout: int = 1):
    output_queue: multiprocessing.Queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=func, args=args + (output_queue,))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        return False
    return output_queue.get()


def _math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = True,
) -> bool:
    if prediction is None or reference is None:
        return False
    if str(str(prediction).strip().lower()) == str(str(reference).strip().lower()):
        return True
    if reference in ["A", "B", "C", "D", "E"] and _choice_answer_clean(str(prediction)) == reference:
        return True

    try:
        if _is_digit(prediction) and _is_digit(reference):
            pred_num = _parse_digits(prediction)
            ref_num = _parse_digits(reference)
            gt_candidates = [ref_num / 100, ref_num, ref_num * 100] if include_percentage else [ref_num]
            for candidate in gt_candidates:
                if candidate is None or pred_num is None:
                    continue
                if (is_close and _numeric_equal(pred_num, candidate)) or (not is_close and pred_num == candidate):
                    return True
            return False
    except Exception:
        pass

    prediction = str(prediction).strip()
    reference = str(reference).strip()
    if not prediction and prediction not in ["0", "False", "false"]:
        return False

    if "pmatrix" in prediction and "pmatrix" not in reference:
        reference = _str_to_pmatrix(reference)

    pred_str, ref_str = prediction, reference
    if (
        prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("(")
    ) or (
        prediction.startswith("(") and prediction.endswith(")") and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for token in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(token, "")
        pred_str = pred_str.replace(token, "")
    if pred_str.lower() == ref_str.lower():
        return True

    if regex_mod.match(r"(\(|\[).+(\)|\])", prediction) and regex_mod.match(r"(\(|\[).+(\)|\])", reference):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(_math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close, timeout) for i in range(len(pred_parts))):
                return True

    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        ref = reference.split("=")
        pred_expr = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref_expr = f"{ref[0].strip()} - ({ref[1].strip()})"
        if _symbolic_equal(pred_expr, ref_expr) or _symbolic_equal(f"-({pred_expr})", ref_expr):
            return True
    elif prediction.count("=") == 1 and len(prediction.split("=")[0].strip()) <= 2 and "=" not in reference:
        if _math_equal(prediction.split("=")[1], reference, include_percentage, is_close, timeout):
            return True
    elif reference.count("=") == 1 and len(reference.split("=")[0].strip()) <= 2 and "=" not in prediction:
        if _math_equal(prediction, reference.split("=")[1], include_percentage, is_close, timeout):
            return True

    if timeout:
        return bool(_call_with_timeout(_symbolic_equal_process, prediction, reference))
    return _symbolic_equal(prediction, reference)


class OmniRuleEvaluator:
    def __init__(self, data_name: str = "omni-math"):
        _ensure_dependencies()
        self.data_name = data_name

    def evaluate(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        judged: List[Dict[str, Any]] = []
        for record in records:
            ground_truth = _parse_ground_truth(str(record["answer"]), self.data_name)
            normalized_prediction = _extract_answer(
                record.get("model_generation", record.get("response", "")),
                self.data_name,
            )
            correctness = _math_equal(normalized_prediction, ground_truth, timeout=True)
            judged.append(
                {
                    **record,
                    "rule_prediction": normalized_prediction,
                    "rule_ground_truth": ground_truth,
                    "correctness": bool(correctness),
                }
            )
        return judged


def get_cached_omni_rule_evaluator(data_name: str = "omni-math") -> OmniRuleEvaluator:
    key = str(data_name)
    evaluator = _OMNI_RULE_CACHE.get(key)
    if evaluator is None:
        evaluator = OmniRuleEvaluator(data_name=key)
        _OMNI_RULE_CACHE[key] = evaluator
    return evaluator
