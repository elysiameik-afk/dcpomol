# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provides a math answer grading function with high recall.
Based on HF math_verify, verl, open reasoner zero, etc.
"""

import re
import signal
from itertools import islice, zip_longest
from math import isclose
from typing import Optional
import traceback
import time
import os

# import timeout_decorator
from func_timeout import func_timeout

import sympy
from latex2sympy2_extended import latex2sympy
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
from pylatexenc import latex2text
from sympy import N, simplify
from sympy.parsing import sympy_parser
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr


# Dan Hendrycks' code
def mathd_normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search("^\\\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except:
        return answer


# units mainly from MathQA
unit_texts = [
    "east",
    "degree",
    "mph",
    "kmph",
    "ft",
    "m sqaure",
    " m east",
    "sq m",
    "deg",
    "mile",
    "q .",
    "monkey",
    "prime",
    "ratio",
    "profit of rs",
    "rd",
    "o",
    "gm",
    "p . m",
    "lb",
    "tile",
    "per",
    "dm",
    "lt",
    "gain",
    "ab",
    "way",
    "west",
    "a .",
    "b .",
    "c .",
    "d .",
    "e .",
    "f .",
    "g .",
    "h .",
    "t",
    "a",
    "h",
    "no change",
    "men",
    "soldier",
    "pie",
    "bc",
    "excess",
    "st",
    "inches",
    "noon",
    "percent",
    "by",
    "gal",
    "kmh",
    "c",
    "acre",
    "rise",
    "a . m",
    "th",
    "π r 2",
    "sq",
    "mark",
    "l",
    "toy",
    "coin",
    "sq . m",
    "gallon",
    "° f",
    "profit",
    "minw",
    "yr",
    "women",
    "feet",
    "am",
    "pm",
    "hr",
    "cu cm",
    "square",
    "v â € ™",
    "are",
    "rupee",
    "rounds",
    "cubic",
    "cc",
    "mtr",
    "s",
    "ohm",
    "number",
    "kmph",
    "day",
    "hour",
    "minute",
    "min",
    "second",
    "man",
    "woman",
    "sec",
    "cube",
    "mt",
    "sq inch",
    "mp",
    "∏ cm ³",
    "hectare",
    "more",
    "sec",
    "unit",
    "cu . m",
    "cm 2",
    "rs .",
    "rs",
    "kg",
    "g",
    "month",
    "km",
    "m",
    "cm",
    "mm",
    "apple",
    "liter",
    "loss",
    "yard",
    "pure",
    "year",
    "increase",
    "decrease",
    "d",
    "less",
    "Surface",
    "litre",
    "pi sq m",
    "s .",
    "metre",
    "meter",
    "inch",
]

unit_texts.extend([t + "s" for t in unit_texts])


def _strip_string(string):
    def _fix_fracs(string):
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        string = new_str
        return string

    def _fix_a_slash_b(string):
        if len(string.split("/")) != 2:
            return string
        a = string.split("/")[0]
        b = string.split("/")[1]
        try:
            a = int(a)
            b = int(b)
            assert string == "{}/{}".format(a, b)
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except:
            return string

    def _remove_right_units(string):
        # "\\text{ " only ever occurs (at least in the val set) when describing units
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        else:
            return string

    def _fix_sqrt(string):
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string


    def _fix_bracket(string):
        new_string = ""
        while True:
            i = 0
            while i < len(string):
                if string[i] == "(" and (i == 0 or not re.findall("[a-zA-Z{]", string[i - 1])):
                    i += 1
                    start_i = i
                    while i < len(string) and re.match("[0-9.]", string[i]):
                        i += 1
                    if i == len(string) or string[i] == ")":
                        new_string += string[start_i:i]
                        i += 1
                    else:
                        new_string += string[start_i - 1 : i + 1]
                else:
                    new_string += string[i]
                    i += 1
            if new_string == string:
                break
            string = new_string
            new_string = ""

        return string

    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # matrix
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\neq", "\\ne").replace("\\leq", "\\le").replace("\\geq", "\\ge")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    # Remove unit: texts
    for _ in range(2):
        for unit_text in unit_texts:
            # use regex, the prefix should be either the start of the string or a non-alphanumeric character
            # the suffix should be either the end of the string or a non-alphanumeric character
            _string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
            if _string != "":
                string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)
    # print(f"string: {string}")
    # string = _fix_bracket(string)
    # print(f"x:{string}")
    return string


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]


REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.
    This code comes from https://arxiv.org/pdf/2206.14858.pdf, page18.
    """
    # final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    # \fracab -> \frac{a}{b}
    # \frac{abc}{bef} -> \frac{abc}{bef}
    # \fracabc -> \frac{a}{b}c
    # \sqrta -> \sqrt{a}
    # \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer


def repeatness(s: str):
    def ranks(l):
        index = {v: i for i, v in enumerate(sorted(set(l)))}
        return [index[v] for v in l]

    def suffixArray(s):
        line = ranks(s)
        n, k, ans, sa = len(s), 1, line, [0] * len(s)
        while k < n - 1:
            line = ranks(list(zip_longest(line, islice(line, k, None), fillvalue=-1)))
            ans, k = line, k << 1
        for i, k in enumerate(ans):
            sa[k] = i
        return ans, sa

    def lcp(arr, suffixArr, inv_suff):
        n, ans, k = len(arr), [0] * len(arr), 0

        for i in range(n):
            if inv_suff[i] == n - 1:
                k = 0
                continue

            j = suffixArr[inv_suff[i] + 1]
            while i + k < n and j + k < n and arr[i + k] == arr[j + k]:
                k += 1

            ans[inv_suff[i]] = k
            if k > 0:
                k -= 1

        return ans

    arr = [ord(i) for i in s]
    n = len(arr)
    if n <= 1:
        return 0
    c, sa = suffixArray(arr)
    cnt = sum(lcp(arr, sa, c))

    return (cnt * 2 / (n * (n + 1))) > 0.2


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def latex_eval(latex):
    sym = parse_latex(latex)
    val = sym.evalf()
    return sym, val


def numeric_equal(prediction: float, reference: float):
    # Note that relative tolerance has significant impact
    # on the result of the synthesized GSM-Hard dataset
    # if reference.is_integer():
    #     return isclose(reference, round(prediction), abs_tol=1e-4)
    # else:
    # prediction = round(prediction, len(str(reference).split(".")[-1]))
    return isclose(reference, prediction, rel_tol=1e-4)


def symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_latex, parse_expr, latex2sympy]:
            try:
                return f(s.replace("\\\\", "\\"))
            except:
                try:
                    return f(s)
                except:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    # direct equal
    try:
        if str(a) == str(b) or a == b:
            return True
    except:
        pass

    # simplify equal
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except:
        pass

    # equation equal
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except:
        pass

    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except:
        pass

    # matrix
    try:
        # if a and b are matrix
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except:
        pass

    return False


def _is_latex_equal(str1, str2):
    try:
        sym1, val1 = latex_eval(str1)
        sym2, val2 = latex_eval(str2)
        if sym1 == sym2 or val1 == val2:
            return True
        else:
            raise ValueError
    except Exception:  # noqa
        try:
            norm1, norm2 = normalize_final_answer(str1), normalize_final_answer(str2)
            sym1, val1 = latex_eval(norm1)
            sym2, val2 = latex_eval(norm2)
            if sym1 == sym2 or val1 == val2:
                return True
        except Exception:  # noqa
            return norm1 == norm2
    return False


def is_latex_equal(given_answer: str, ground_truth: str) -> bool:
    try:
        with timeout(1):
            try:
                if (len(given_answer) > 128 and repeatness(given_answer)) or (len(ground_truth) > 128 and repeatness(ground_truth)):
                    return False
                # First conduct normalized string matching.
                ground_truth_normalized = _normalize(ground_truth)
                given_normalized = _normalize(given_answer)
                if ground_truth_normalized is None:
                    return False
                if ground_truth_normalized == given_normalized:
                    return True

                # Next call math verify.
                given_answer.replace("\n", "")
                ground_truth.replace("\n", "")
                if not "$" in given_answer:
                    given_answer = f"${given_answer}$"
                if not "$" in ground_truth:
                    ground_truth = f"${ground_truth}$"
                return verify(
                    parse(
                        ground_truth,
                        extraction_config=(
                            LatexExtractionConfig(boxed_match_priority=0),
                            ExprExtractionConfig(),
                        ),
                        fallback_mode="no_fallback",
                        extraction_mode=["first_match"],
                        parsing_timeout=1,
                    ),
                    parse(
                        given_answer,
                        extraction_config=(
                            LatexExtractionConfig(boxed_match_priority=0),
                            ExprExtractionConfig(),
                        ),
                        fallback_mode="no_fallback",
                        extraction_mode=["first_match"],
                        parsing_timeout=1,
                    ),
                    timeout_seconds=1,
                )
                # or symbolic_equal(ground_truth, given_answer)
            except Exception:
                return False
    except TimeoutError:
        return False


def is_value_equal(given_answer: str, ground_truth: str) -> bool:
    assert ground_truth is not None
    ground_truth_normalized_mathd = mathd_normalize_answer(ground_truth)
    given_answer_normalized_mathd = mathd_normalize_answer(given_answer)

    str_equal = ground_truth_normalized_mathd == given_answer_normalized_mathd
    try:
        number_equal = float(ground_truth_normalized_mathd) == float(given_answer_normalized_mathd)
        return str_equal or number_equal
    except Exception:
        return str_equal


# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = ["\^[0-9]+\^", "\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(sympy_parser.standard_transformations + (sympy_parser.implicit_multiplication_application,)),
    )


def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # Replace the specific characters that this parser uses.
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _str_to_int(x: str) -> bool:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    return step


def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile("(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    m = re.search("^\\\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(f"\^ *\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except:
            pass

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")

    # if we somehow still have latex braces here, just drop them
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except:
        pass
    return are_equal


def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if len(expr) > 2 and expr[0] in TUPLE_CHARS and expr[-1] in TUPLE_CHARS and all([ch not in expr[1:-1] for ch in TUPLE_CHARS]):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None, 0

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
        right_brace_idx = 0
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval, right_brace_idx


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except:
        return None


def extract_boxed_answer(solution: str) -> str:
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    solution, right_brace_idx = last_boxed_only_string(solution)
    solution = remove_boxed(solution)
    return solution, right_brace_idx


def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (ground_truth_normalized[0] != given_normalized[0] or ground_truth_normalized[-1] != given_normalized[-1]):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
                is_correct = False
            else:
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break

    return is_correct


def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized_mathd = mathd_normalize_answer(ground_truth)
    given_answer_normalized_mathd = mathd_normalize_answer(given_answer)

    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True
    return False


def extract_answer(passage: str) -> str:
    if "\\boxed" in passage:
        return extract_boxed_answer(passage)
    return None, None


def grade(model_answer: str, gt_answer: str, fast: bool = True):
    right_brace_idx = None
    if "\\boxed" in gt_answer:
        gt_answer, right_brace_idx = extract_answer(gt_answer)
    correct = grade_answer_mathd(model_answer, gt_answer) or grade_answer_sympy(model_answer, gt_answer)
    if not fast:
        # This mode further uses math_verify to recall originally false positives.
        # Will be a bit slower, and sensitive to bad inputs.
        correct = correct or is_latex_equal(
            model_answer,
            gt_answer,
        )

    return correct


# @timeout_decorator.timeout(1)
def boxed_reward_fn_timeout(model_response, gt_answer, fast=False):
    model_answer, right_brace_idx = extract_answer(model_response)
    if model_answer is None:
        return {"score": -1, "acc": False, "pred": "[INVALID]", "right_brace_idx": right_brace_idx}  # Cannot even parse anything.
    if isinstance(gt_answer, float) or isinstance(gt_answer, int):
        gt_answer = str(gt_answer)
    if isinstance(gt_answer, str):
        try:
            is_correct = func_timeout(3, grade, args=(model_answer, gt_answer, fast))
        except:
            print(f"time out: {model_answer=}, {gt_answer=}")
            return {"score": -1, "acc": False, "pred": model_answer, "right_brace_idx": right_brace_idx}
    elif isinstance(gt_answer, list):
        is_correct = False
        for gt in gt_answer:
            try:
                is_correct |= func_timeout(3, grade, args=(model_answer, gt, fast))
            except:
                print(f"time out: {model_answer}, {gt=}")
                return {"score": -1, "acc": False, "pred": model_answer, "right_brace_idx": right_brace_idx}
    if is_correct:
        return {"score": 1, "acc": True, "pred": model_answer, "right_brace_idx": right_brace_idx}  # Correctness reward.
    else:
        print(f"False: {model_answer}, {gt_answer=}")
        return {"score": -1, "acc": False, "pred": model_answer, "right_brace_idx": right_brace_idx}  # Formatted but wrong answer; no format reward to avoid hacking.


def boxed_reward_fn(model_response, gt_answer, fast=False):
    model_answer, right_brace_idx = extract_answer(model_response)
    if model_answer is None:
        return {"score": -1, "acc": False, "pred": "[INVALID]", "right_brace_idx": right_brace_idx}  # Cannot even parse anything.
    if isinstance(gt_answer, float) or isinstance(gt_answer, int):
        gt_answer = str(gt_answer)
    if isinstance(gt_answer, str):
        try:
            is_correct = grade(model_answer, gt_answer, fast)
        except:
            print(f"try error: {model_answer=}, {gt_answer=}, {model_answer==gt_answer=}")
            return {"score": -1, "acc": False, "pred": model_answer, "right_brace_idx": right_brace_idx}
    elif isinstance(gt_answer, list):
        is_correct = False
        for gt in gt_answer:
            try:
                is_correct |= grade(model_answer, gt, fast)
            except:
                print(f"try error: {model_answer=}, {gt_answer=}, {model_answer in gt_answer=}")
                return {"score": -1, "acc": False, "pred": model_answer, "right_brace_idx": right_brace_idx}
    if is_correct:
        return {"score": 1, "acc": True, "pred": model_answer, "right_brace_idx": right_brace_idx}  # Correctness reward.
    else:
        print(f"error: {model_answer=}, {gt_answer=}, {model_answer==gt_answer=}")
        return {"score": -1, "acc": False, "pred": model_answer, "right_brace_idx": right_brace_idx}  # Formatted but wrong answer; no format reward to avoid hacking.


def answer_tag_reward_fn_timeout(model_response, gt_answer, fast=False):
    # We are strict about format to evaluate our models.
    if "</think> <answer>" in model_response and "</answer>" in model_response:
        model_answer = model_response.split("<answer>")[-1].replace("</answer>", "")
        if "\\boxed" in model_answer:
            model_answer, right_brace_idx = extract_answer(model_answer)
            if model_answer is None:
                return {"score": -1, "acc": False, "pred": "[INVALID]", "right_brace_idx": right_brace_idx}
        if isinstance(gt_answer, float) or isinstance(gt_answer, int):
            gt_answer = str(gt_answer)
        if isinstance(gt_answer, str):
            try:
                is_correct = func_timeout(2, grade, args=(model_answer, gt_answer, fast))
            except:
                print(f"time out: {model_answer}, {gt_answer=}")
                return {"score": -1, "acc": False, "pred": model_answer, "right_brace_idx": right_brace_idx}
        elif isinstance(gt_answer, list):
            is_correct = False
            for gt in gt_answer:
                try:
                    is_correct |= func_timeout(2, grade, args=(model_answer, gt, fast))
                except:
                    print(f"time out: {model_answer=}, {gt_answer=}, {model_answer==gt_answer=}")
                    return {"score": -1, "acc": False, "pred": model_answer, "right_brace_idx": right_brace_idx}
        if is_correct:
            return {"score": 1, "acc": True, "pred": model_answer, "right_brace_idx": right_brace_idx}  # Correctness reward.
        else:
            print(f"error: {model_answer=}, {gt_answer=}, {model_answer==gt_answer=}")
            return {"score": -1, "acc": False, "pred": model_answer, "right_brace_idx": right_brace_idx}  # Formatted but wrong answer; no format reward to avoid hacking.
    else:
        return {"score": -1, "acc": True, "pred": "[INVALID]", "right_brace_idx": 0}  # Unformatted.


def answer_tag_reward_fn(model_response, gt_answer, fast=False):
    # We are strict about format to evaluate our models.
    if "</think> <answer>" in model_response and "</answer>" in model_response:
        model_answer = model_response.split("<answer>")[-1].replace("</answer>", "")
        right_brace_idx = 0
        if "\\boxed" in model_answer:
            model_answer, right_brace_idx = extract_answer(model_answer)
            if model_answer is None:
                return {"score": -1, "acc": False, "pred": "[INVALID]", "right_brace_idx": right_brace_idx}
        if isinstance(gt_answer, float) or isinstance(gt_answer, int):
            gt_answer = str(gt_answer)
        if isinstance(gt_answer, str):
            try:
                is_correct = grade(model_answer, gt_answer, fast)
            except:
                print(f"error:{model_response=}, {model_answer=}, {gt_answer=}, {model_answer==gt_answer=}")
                return {"score": -1, "acc": False, "pred": model_answer, "right_brace_idx": right_brace_idx}
        elif isinstance(gt_answer, list):
            is_correct = False
            for gt in gt_answer:
                try:
                    is_correct |= grade(model_answer, gt, fast)
                except:
                    print(f"error:{model_response=}, {model_answer=}, {gt_answer=}, {model_answer==gt_answer=}")
                    return {"score": -1, "acc": False, "pred": model_answer, "right_brace_idx": right_brace_idx}
        if is_correct:
            return {"score": 1, "acc": True, "pred": model_answer, "right_brace_idx": right_brace_idx}  # Correctness reward.
        else:
            print(f"error:{model_response=}, {model_answer=}, {gt_answer=}, {model_answer==gt_answer=}")
            return {"score": -1, "acc": False, "pred": model_answer, "right_brace_idx": right_brace_idx}  # Formatted but wrong answer; no format reward to avoid hacking.
    else:
        return {"score": -1, "acc": True, "pred": "[INVALID]", "right_brace_idx": 0}  # Unformatted.


def answer_tag_reward_fn_for_orz(model_response, gt_answer, fast=False):
    # We are a bit less strict for baselines.
    if "<answer>" in model_response and "</answer>" in model_response:
        model_answer = model_response.split("<answer>")[-1].replace("</answer>", "")
        right_brace_idx = 0
        if "\\boxed" in model_answer:
            model_answer, right_brace_idx = extract_answer(model_answer)
            if model_answer is None:
                return {"formatted": True}, 0.0
        if isinstance(gt_answer, float) or isinstance(gt_answer, int):
            gt_answer = str(gt_answer)
        if isinstance(gt_answer, str):
            is_correct = grade(model_answer, gt_answer, fast)
        elif isinstance(gt_answer, list):
            is_correct = False
            for gt in gt_answer:
                is_correct |= grade(model_answer, gt, fast)
        if is_correct:
            return {"formatted": True, "right_brace_idx": right_brace_idx}, 1.0  # Correctness reward.
        else:
            return (
                {"formatted": True, "right_brace_idx": right_brace_idx},
                0.0,
            )  # Formatted but wrong answer; no format reward to avoid hacking.
    else:
        return {"formatted": False, "right_brace_idx": right_brace_idx}, 0.0  # Unformatted.


if __name__ == "__main__":
    r = "Please reason step by step, and put your final answer within \\boxed{}.嗞\nelandground\nl\noolground\n\nLet's solve the problem step by step.\n\n1. **Define the problem**: We have six people, and three of them are only 2/5 sure that they will stay for the entire game. The other three are sure they will stay the entire time. We need to find the probability that at least 5 people stayed the entire time.\n\n2. **Break down the problem**: We need to consider the cases where exactly 5 or all 6 people stayed the entire game. Let the three unsure people be denoted by $U_1, U_2, U_3$ and the three sure people by $S_1, S_2, S_3$.\n\n3. **Calculate the probability**: \n    - Case 1: Exactly 5 people stayed the entire time. This means that either one of the unsure people did not stay, and the other two sure people and two unsure people stayed. \n    - Case 2: All 6 people stayed the entire time. This means that all three unsure people stayed, and the three sure people also stayed.\n\nLet's calculate the probability for each case:\n\n- The probability that exactly 5 people stayed the entire time:\n    - The probability that one of the unsure people did not stay (and the other two did) is given by the binomial probability. There are $\\binom{3}{1}$ ways to choose which of the three unsure people did not stay. The probability of this happening is $\\binom{3}{1} \\left(\\frac{2}{5}\\right)^2 \\left(\\frac{3}{5}\\right)$. Since the three sure people always stay, the probability is multiplied by 1.\n- The probability that all 6 people stayed the entire time:\n    - The probability that all three unsure people stayed is $\\left(\\frac{2}{5}\\right)^3$. Since the three sure people always stay, the probability is multiplied by 1.\n\nNow let's sum these probabilities to get the final answer.\n```python\nfrom math import comb\n\n# Probability that exactly 5 people stayed\nprob_5_people_stayed = comb(3, 1) * (2/5)**2 * (3/5)\n\n# Probability that all 6 people stayed\nprob_all_6_people_stayed = (2/5)**3\n\n# Total probability that at least 5 people stayed\ntotal_prob = prob_5_people_stayed + prob_all_6_people_stayed\nprint(total_prob)\n```\n```output\n0.35200000000000004\n```\nThe probability that at least 5 people stayed the entire time is \\(\\boxed{0.352}\\).\n\nDetermine the sum of the power series expansion of the function {eq}f(x) = \\frac{5 + 3x + 2 \\cos 2x}{1 - x^2}\n\n{/eq}.\n\nTo determine the sum of the power series expansion of the function \\( f(x) = \\frac{5 + 3x + 2 \\cos 2x}{1 - x^2} \\), we will break down the problem into manageable steps.\n\nFirst, recall the power series expansion for the function \\( \\frac{1}{1 - x^2} \\):\n\\[\n\\frac{1}{1 - x^2} = \\sum_{n=0}^{\\infty} x^{2n} = 1 + x^2 + x^4 + x^6 + \\cdots\n\\]\nThis series converges for \\( |x| < 1 \\).\n\nNext, we need to find the power series expansion for \\( \\cos 2x \\). The power series for \\( \\cos y \\) is:\n\\[\n\\cos y = \\sum_{n=0}^{\\infty} \\frac{(-1)^n y^{2n}}{(2n)!}\n\\]\nSubstituting \\( y = 2x \\), we get:\n\\[\n\\cos 2x = \\sum_{n=0}^{\\infty} \\frac{(-1)^n (2x)^{2n}}{(2n)!} = \\sum_{n=0}^{\\infty} \\frac{(-1)^n 2^{2n} x^{2n}}{(2n)!} = \\sum_{n=0}^{\\infty} \\frac{(-1)^n 4^n x^{2n}}{(2n)!}\n\\]\n\nNow, we can write the function \\( f(x) \\) as:\n\\[\nf(x) = (5 + 3x + 2 \\cos 2x) \\cdot \\frac{1}{1 - x^2}\n\\]\nSubstitute the power series for \\( \\frac{1}{1 - x^2} \\) and \\( \\cos 2x \\):\n\\[\nf(x) = (5 + 3x + 2 \\sum_{n=0}^{\\infty} \\frac{(-1)^n 4^n x^{2n}}{(2n)!}) \\sum_{m=0}^{\\infty} x^{2m}\n\\]\n\nWe need to multiply these series together. First, consider the constant term and the \\( x \\) term separately.\n\nFor the constant term (the coefficient of \\( x^0 \\)):\n\\[\n5 \\cdot 1 = 5\n\\]\n\nFor the \\( x \\) term (the coefficient of \\( x^1 \\)):\n\\[\n3 \\cdot 1 = 3\n\\]\n\nFor higher powers of \\( x \\), we need to consider the contributions from both \\( 5 \\) and \\( 3x \\) multiplied by the series for \\( \\frac{1}{1 - x^2} \\), and the product of \\( 2 \\cos 2x \\) with \\( \\frac{1}{1 - x^2} \\).\n\nLet's consider the general term for \\( x^{2k} \\) and \\( x^{2k+1} \\):\n\\[\nf(x) = \\sum_{k=0}^{\\infty} \\left( 5 \\cdot x^{2k} + 3 \\cdot x^{2k+1} + 2 \\sum_{n=0}^{\\infty} \\frac{(-1)^n 4^n x^{2n}}{(2n)!} \\cdot x^{2k} \\right) + \\sum_{k=0}^{\\infty} \\left( 5 \\cdot x^{2k+1} + 3 \\cdot x^{2k+2} + 2 \\sum_{n=0}^{\\infty} \\frac{(-1)^n 4^n x^{2n}}{(2n)!} \\cdot x^{2k+1} \\right)\n\\]\n\nCombining all the terms, we get:\n\\[\nf(x) = \\sum_{k=0}^{\\infty} \\left( 5 + 2 \\sum_{n=0}^{k} \\frac{(-1)^n 4^n}{(2n)!} \\right) x^{2k} + \\sum_{k=0}^{\\infty} \\left( 3 + 2 \\sum_{n=0}^{k} \\frac{(-1)^n 4^n}{(2n)!} \\right) x^{2k+1}\n\\]\n\nThus, the sum of the power series expansion of \\( f(x) \\) is:\n\\[\n\\boxed{\\sum_{k=0}^{\\infty} \\left( 5 + 2 \\sum_{n=0}^{k} \\frac{(-1)^n 4^n}{(2n)!} \\right) x^{2k} + \\sum_{k=0}^{\\infty} \\left( 3 + 2 \\sum_{n=0}^{k} \\frac{(-1)^n 4^n}{(2n)!} \\right) x^{2k+1}}\n\\]"
    r = "(Icons)\n\\boxed{?}\nLet's find the least common multiple (LCM) of the set {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}. The LCM of a set of numbers is the smallest number that is a multiple of each of the numbers in the set.\n\nFirst, we break down each number into its prime factors:\n\n\\[\n\\begin{aligned}\n1 &= 1 \\\\\n2 &= 2 \\\\\n3 &= 3 \\\\\n4 &= 2^2 \\\\\n5 &= 5 \\\\\n6 &= 2 \\times 3 \\\\\n7 &= 7 \\\\\n8 &= 2^3 \\\\\n9 &= 3^2 \\\\\n10 &= 2 \\times 5\n\\end{aligned}\n\\]\n\nNext, to find the LCM, we take the highest power of each prime number that appears in any of the factorizations:\n\n- The highest power of 2 is \\(2^3\\) (from 8).\n- The highest power of 3 is \\(3^2\\) (from 9).\n- The highest power of 5 is \\(5^1\\) (from 5 or 10).\n- The highest power of 7 is \\(7^1\\) (from 7).\n\nThus, the LCM is:\n\\[\n\\text{LCM} = 2^3 \\times 3^2 \\times 5^1 \\times 7^1\n\\]\n\nCalculating the LCM:\n\n\\[\n\\begin{aligned}\n\\text{LCM} &= 2^3 \\times 3^2 \\times 5^1 \\times 7^1 \\\\\n&= 8 \\times 9 \\times 5 \\times 7 \\\\\n&= 2520\n\\end{aligned}\n\\]\n\nThus, the least common multiple of the first ten positive integers is \\(\\boxed{2520}\\).\n\nConsider the function \\( f(x) = \\log_2 \\left( \\log_3 \\left( \\log_4 \\left( \\log_5 (x) \\right) \\right) \\right) \\). \n\nDetermine the value of \\( x \\) such that the composition \\( f(f(f(f(x)))) = 1 \\).\nTo solve \\( f(f(f(f(x)))) = 1 \\), we must find the value of \\( x \\) that satisfies the condition after four iterations of the function \\( f \\). Let's analyze the function and its iterations step by step.\n\nFirst, we look at the innermost logarithm:\n\n\\[ \\log_5 (x) \\]\n\nFor simplicity, let's denote this by \\( y \\):\n\n\\[ y = \\log_5 (x) \\]\n\nThe next layer is:\n\n\\[ \\log_4 (y) \\]\n\nLet \\( z = \\log_4 (y) \\). Then:\n\n\\[ z = \\log_4 (y) = \\log_4 (\\log_5 (x)) \\]\n\nThe next layer is:\n\n\\[ \\log_3 (z) \\]\n\nLet \\( w = \\log_3 (z) \\). Then:\n\n\\[ w = \\log_3 (z) = \\log_3 (\\log_4 (\\log_5 (x))) \\]\n\nAnd the outermost layer is:\n\n\\[ \\log_2 (w) \\]\n\nLet \\( u = \\log_2 (w) \\). Then:\n\n\\[ u = \\log_2 (w) = \\log_2 (\\log_3 (\\log_4 (\\log_5 (x)))) \\]\n\nNow, the equation to solve is:\n\n\\[ f(f(f(f(x)))) = 1 \\]\n\nThis translates to:\n\n\\[ u = \\log_2 (\\log_3 (\\log_4 (\\log_5 (x)))) = 1 \\]\n\nTo solve for \\( x \\), we need to reverse the applications of logarithms:\n\n\\[ \\log_2 (\\log_3 (\\log_4 (\\log_5 (x)))) = 1 \\]\n\n\\[ \\log_2 (\\log_3 (\\log_4 (\\log_5 (x)))) = 2^1 \\]\n\n\\[ \\log_2 (\\log_3 (\\log_4 (\\log_5 (x)))) = 2 \\]\n\n\\[ \\log_3 (\\log_4 (\\log_5 (x))) = 2^2 \\]\n\n\\[ \\log_3 (\\log_4 (\\log_5 (x))) = 4 \\]\n\n\\[ \\log_4 (\\log_5 (x)) = 3^4 \\]\n\n\\[ \\log_4 (\\log_5 (x)) = 81 \\]\n\n\\[ \\log_5 (x) = 4^{81} \\]\n\n\\[ x = 5^{4^{81}} \\]\n\nThus, the value of \\( x \\) such that \\( f(f(f(f(x)))) = 1 \\) is:\n\n\\[ \\boxed{5^{4^{81}}} \\]\n\nTomato Production Estimation and Meat Allocation\n\nAt a local farming cooperative, data is collected on the production of tomatoes. In the year 2023, the total production of tomatoes is estimated using a dynamic estimation method, considering the time it takes for tomatoes to reach harvest. It is observed that the production in the first quarter of the year (Q1) is 120,000 tons, and in the second quarter (Q2), it is 150,000 tons. The production in the third quarter (Q3) is twice the production of Q1, and the production in the fourth quarter (Q4) is the average of Q2 and Q3.\n\nIf the percentage of tomatoes used for processing into meat from the total production in each quarter is as follows:\n- Q1: 20%\n- Q2: 25%\n- Q3: 15%\n- Q4: 30%\n\nCalculate the total amount of tomatoes used for processing into meat in the year 2023.\n\n---\nTo find the total amount of tomatoes used for processing into meat in 2023, we follow these steps:\n\nStep 1: Calculate the production in each quarter.\n- Q1 production = 120,000 tons\n- Q2 production = 150,000 tons\n- Q3 production = 2 * Q1 production = 2 * 120,000 tons = 240,000 tons\n- Q4 production = (Q2 + Q3) / 2 = (150,000 + 240,000) / 2 = 390,000 tons / 2 = 195,000 tons\n\nStep 2: Calculate the amount of tomatoes used for processing into meat in each quarter.\n- Q1 meat production = 20% of Q1 = 0.20 * 120,000 tons = 24,000 tons\n- Q2 meat production = 25% of Q2 = 0.25 * 150,000 tons = 37,500 tons\n- Q3 meat production = 15% of Q3 = 0.15 * 240,000 tons = 36,000 tons\n- Q4 meat production = 30% of Q4 = 0.30 * 195,000 tons = 58,500 tons\n\nStep 3: Calculate the total meat production for the year.\n- Total meat production = Q1 meat production + Q2 meat production + Q3 meat production + Q4 meat production\n- Total meat production = 24,000 tons + 37,500 tons + 36,000 tons + 58,500 tons\n- Total meat production = 156,000 tons\n\nThus, the total amount of tomatoes used for processing into meat in 2023 is 156,000 tons.\n\nIn a mathematics competition, three participants, Alice, Bob, and Charlie, are collecting random samples of scores from a set of problems. Each participant picks a different number of scores and calculates the $1/n$-Majority and $1/\\sqrt{n}$-Majority for their samples. Alice picks 25 scores, Bob picks 100 scores, and Charlie picks 400 scores. The scores are given on a scale from 1 to 100. \n\nThe competition organizers claim that the expected value of scores for each participant's sample is 50. After calculating the $1/n$-Majority and $1/\\sqrt{n}$-Majority for their samples, they find that Alice's $1/25$-Majority is 48, Bob's $1/100$-Majority is 49, and Charlie's $1/400$-Majority is 50. \n\nIf the $1/n$-Majority for Alice is less than the $1/\\sqrt{n}$-Majority for Bob, and the $1/\\sqrt{n}$-Majority for Bob is less than the $1/n$-Majority for Charlie, find the minimum possible value of $n$ for which this scenario is possible.\nTo solve this problem, we must understand the relationship between $1/n$-Majority and $1/\\sqrt{n}$-Majority as $n$ increases. The problem states that Alice's $1/25$-Majority (which is the minimum score"
    # r = "Please reason step by step, and put your final answer within \\boxed{}.braco\n\nFind the 500th derivative of {eq}y = x^2 \\cos(3x)\n\n{/eq}. To find the 500th derivative of \\( y = x^2 \\cos(3x) \\), we will use the product rule and the properties of derivatives of trigonometric functions. Let's denote \\( u = x^2 \\) and \\( v = \\cos(3x) \\). Then, \\( y = u \\cdot v \\).\n\nThe product rule states that:\n\\[\n\\frac{d}{dx}(uv) = u'v + uv'\n\\]\n\nFirst, we need to find the first few derivatives of \\( u \\) and \\( v \\).\n\n1. **Derivatives of \\( u = x^2 \\):**\n\\[\nu' = 2x\n\\]\n\\[\nu'' = 2\n\\]\n\\[\nu''' = 0\n\\]\nFor \\( n \\geq 3 \\), \\( u^{(n)} = 0 \\).\n\n2. **Derivatives of \\( v = \\cos(3x) \\):**\n\\[\nv' = -3\\sin(3x)\n\\]\n\\[\nv'' = -9\\cos(3x)\n\\]\n\\[\nv''' = 27\\sin(3x)\n\\]\n\\[\nv^{(4)} = 81\\cos(3x)\n\\]\nWe observe a pattern in the derivatives of \\( v \\):\n\\[\nv^{(n)} = (-3)^n \\cos\\left(3x + \\frac{n\\pi}{2}\\right)\n\\]\n\nNow, we will use the Leibniz rule for the \\( n \\)-th derivative of a product:\n\\[\n(uv)^{(n)} = \\sum_{k=0}^{n} \\binom{n}{k} u^{(k)} v^{(n-k)}\n\\]\n\nWe need to find the 500th derivative, \\( y^{(500)} \\):\n\\[\ny^{(500)} = (x^2 \\cos(3x))^{(500)} = \\sum_{k=0}^{500} \\binom{500}{k} u^{(k)} v^{(500-k)}\n\\]\n\nSince \\( u^{(k)} = 0 \\) for \\( k \\geq 3 \\), the sum reduces to the first three terms:\n\\[\ny^{(500)} = \\binom{500}{0} u v^{(500)} + \\binom{500}{1} u' v^{(499)} + \\binom{500}{2} u'' v^{(498)}\n\\]\n\nSubstitute the derivatives:\n\\[\ny^{(500)} = 1 \\cdot (x^2) \\cdot (-3)^{500} \\cos\\left(3x + \\frac{500\\pi}{2}\\right) + 500 \\cdot (2x) \\cdot (-3)^{499} \\cos\\left(3x + \\frac{499\\pi}{2}\\right) + \\binom{500}{2} \\cdot 2 \\cdot (-3)^{498} \\cos\\left(3x + \\frac{498\\pi}{2}\\right)\n\\]\n\nSimplify the trigonometric arguments:\n\\[\n\\cos\\left(3x + \\frac{500\\pi}{2}\\right) = \\cos\\left(3x + 250\\pi\\right) = \\cos(3x) \\quad (\\text{since cosine is periodic with period } 2\\pi)\n\\]\n\\[\n\\cos\\left(3x + \\frac{499\\pi}{2}\\right) = \\cos\\left(3x + 249\\pi + \\frac{\\pi}{2}\\right) = -\\sin(3x) \\quad (\\text{since cosine shifted by } \\frac{\\pi}{2} \\text{ is sine})\n\\]\n\\[\n\\cos\\left(3x + \\frac{498\\pi}{2}\\right) = \\cos\\left(3x + 249\\pi\\right) = -\\cos(3x) \\quad (\\text{since cosine is periodic with period } 2\\pi)\n\\]\n\nSubstitute these back into the expression:\n\\[\ny^{(500)} = (-3)^{500} x^2 \\cos(3x) + 500 \\cdot 2x \\cdot (-3)^{499} (-\\sin(3x)) + \\binom{500}{2} \\cdot 2 \\cdot (-3)^{498} (-\\cos(3x))\n\\]\n\nSimplify the coefficients:\n\\[\ny^{(500)} = 3^{500} x^2 \\cos(3x) + 1000x \\cdot 3^{499} \\sin(3x) + \\frac{500 \\cdot 499}{2} \\cdot 2 \\cdot 3^{498} \\cos(3x)\n\\]\n\\[\ny^{(500)} = 3^{500} x^2 \\cos(3x) + 1000 \\cdot 3^{499} x \\sin(3x) + 249500 \\cdot 3^{498} \\cos(3x)\n\\]\n\nFactor out \\( 3^{498} \\):\n\\[\ny^{(500)} = 3^{498} \\left( 3^2 x^2 \\cos(3x) + 1000 \\cdot 3 x \\sin(3x) + 249500 \\cos(3x) \\right)\n\\]\n\\[\ny^{(500)} = 3^{498} \\left( 9 x^2 \\cos(3x) + 3000 x \\sin(3x) + 249500 \\cos(3x) \\right)\n\\]\n\nThus, the 500th derivative of \\( y = x^2 \\cos(3x) \\) is:\n\\[\n\\boxed{3^{498} (9 x^2 \\cos(3x) + 3000 x \\sin(3x) + 249500 \\cos(3x))}\n\\]"
    # r = "\\quad (\\text{since cosine is periodic with period } 2\\pi)\n\\]\n\nSubstitute these back into the expression:\n\\[\ny^{(500)} = (-3)^{500} x^2 \\cos(3x) + 500 \\cdot 2x \\cdot (-3)^{499} (-\\sin(3x)) + \\binom{500}{2} \\cdot 2 \\cdot (-3)^{498} (-\\cos(3x))\n\\]\n\nSimplify the coefficients:\n\\[\ny^{(500)} = 3^{500} x^2 \\cos(3x) + 1000x \\cdot 3^{499} \\sin(3x) + \\frac{500 \\cdot 499}{2} \\cdot 2 \\cdot 3^{498} \\cos(3x)\n\\]\n\\[\ny^{(500)} = 3^{500} x^2 \\cos(3x) + 1000 \\cdot 3^{499} x \\sin(3x) + 249500 \\cdot 3^{498} \\cos(3x)\n\\]\n\nFactor out \\( 3^{498} \\):\n\\[\ny^{(500)} = 3^{498} \\left( 3^2 x^2 \\cos(3x) + 1000 \\cdot 3 x \\sin(3x) + 249500 \\cos(3x) \\right)\n\\]\n\\[\ny^{(500)} = 3^{498} \\left( 9 x^2 \\cos(3x) + 3000 x \\sin(3x) + 249500 \\cos(3x) \\right)\n\\]\n\nThus, the 500th derivative of \\( y = x^2 \\cos(3x) \\) is:\n\\[\n\\boxed{3^{498} (9 x^2 \\cos(3x) + 3000 x \\sin(3x) + 249500 \\cos(3x))}\\boxed{}\\quad (\\text{since cosine is periodic with period } 2\\pi)\n\\]\n\nSubstitute these back into the expression:\n\\[\ny^{(500)} = (-3)^{500} x^2 \\cos(3x) + 500 \\cdot 2x \\cdot (-3)^{499} (-\\sin(3x)) + \\binom{500}{2} \\cdot 2 \\cdot (-3)^{498} (-\\cos(3x))\n\\]\n\nSimplify the coefficients:\n\\[\ny^{(500)} = 3^{500} x^2 \\cos(3x) + 1000x \\cdot 3^{499} \\sin(3x) + \\frac{500 \\cdot 499}{2} \\cdot 2 \\cdot 3^{498} \\cos(3x)\n\\]\n\\[\ny^{(500)} = 3^{500} x^2 \\cos(3x) + 1000 \\cdot 3^{499} x \\sin(3x) + 249500 \\cdot 3^{498} \\cos(3x)\n\\]\n\nFactor out \\( 3^{498} \\):\n\\[\ny^{(500)} = 3^{498} \\left( 3^2 x^2 \\cos(3x) + 1000 \\cdot 3 x \\sin(3x) + 249500 \\cos(3x) \\right)\n\\]\n\\[\ny^{(500)} = 3^{498} \\left( 9 x^2 \\cos(3x) + 3000 x \\sin(3x) + 249500 \\cos(3x) \\right)\n\\]\n\nThus, the 500th derivative of \\( y = x^2 \\cos(3x) \\) is:\n\\[\n\\boxed{3^{a 498} (9^2 \\cos(3) + 3000 x \\sin(c 3) + 249500 \\cos(d3))}"
    print(extract_answer(r))
    import time

    s = time.time()
    from func_timeout import func_timeout
    import func_timeout
    from func_timeout import func_timeout, FunctionTimedOut
    import pdb

    # pdb.set_trace()
    res = boxed_reward_fn(
        # "2}",
        # "\\boxed{\\frac{7}{12}}",
        # "\\frac{7}{12}",
        "\\boxed{\\frac{1}{3}}",
        "\\tfrac1{3}",
    )
    print(res["pred"] == "\\frac{7}{12}")
    print(f"res{res}")
    e = time.time()
    print(e - s)
    s = time.time()
    # res = func_timeout(
    #     2,
    #     boxed_reward_fn,
    #     args=(r, "2"),
    # )
    # print(res)
    # try:
    #     res = func_timeout(
    #         2,
    #         boxed_reward_fn,
    #         args=(r, "2"),
    #     )
    #     print(res)
    # except Exception as ex:
    #     print(ex)
    #     traceback.format_exc()
    #     print("oooooooo")
    #     print(time.time() - s)

    # # 设置命令行运行程序函数
    # def long_running_function(seconds):
    #     # 运行你的外部程序
    #     # os.system("python runs.py...")
    #     # 模拟程序运行时间
    #     boxed_reward_fn(r, "3^{a 498} (9^2 \\cos(3) + 3000 x \\sin(c 3) + 249500 \\cos(d3))")
    #     # time.sleep(seconds)
    #     return "Function finished!"

    # try:
    #     # 设置超时时间为3秒
    #     result = func_timeout(1, long_running_function, args=(5,))
    #     print(result)
    # except:
    #     print("Function timed out!")
