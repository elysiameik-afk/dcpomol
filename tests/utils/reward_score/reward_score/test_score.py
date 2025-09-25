import sys
import json
from glob import glob

from verl.utils.reward_score import default_compute_score

from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Lock
import traceback
import pdb

lock = Lock()
print("start")


def get_acc(acc_writer, line):

    jt = json.loads(line.strip())
    ground_truth = eval(jt["reward_model"])["ground_truth"]
    # if len(jt["qwen2.5-72b-instruct"]) == 0:
    #     print(file, jt["prompt"][0]["content"])
    try:
        result = eval(jt["qwen2.5-72b-instruct"])
        acc_result = [False for _ in range(len(result))]
        pred = [[] for _ in range(len(result))]
        for i in range(len(result)):
            solution_str = result[i]
            acc = False
            for data_source in ["math_dapo", "qwen_qen"]:
                res = default_compute_score(data_source, solution_str, ground_truth)
                acc |= res["acc"]
                pred[i].append(res["pred"])
            acc_result[i] = acc
        jt["acc"] = acc_result
        jt["pred"] = pred

        # with lock:
        #     acc_writer.write(json.dumps(jt, ensure_ascii=False) + "\n")
        #     acc_writer.flush()
    except Exception as e:
        # pdb.set_trace()
        traceback.format_exc()
        pass


# 并发处理
# for file in glob("/yangshihui/data/RL/infer_result/*.jsonl"):
#     acc_writer = None  # open(f"{file}.acc", "w")
#     print(file)
#     with open(file) as f:
#         with ThreadPoolExecutor(max_workers=50) as executor:
#             futures = [executor.submit(get_acc, acc_writer, line) for line in f]
#             for future in as_completed(futures):
#                 future.result()

# 测试

# for file in glob("/global_data/med/yangsh/data/RL/infer_result/*.jsonl"):
#     acc_writer = None  # open(f"{file}.acc", "w")
#     print(file)
#     with open(file) as f:
#         for line in f:
#             get_acc(None, line)

data_source = "qwen_dapo17k"
response_str = "To find the sum of the real roots of the polynomial \\(4x^4 - 3x^2 + 7x - 3 = 0\\), we start by considering the polynomial and its properties. One strategy is to test for possible rational roots using the Rational Root Theorem, which states that any rational root of the polynomial \\(a_n x^n + a_{n-1} x^{n-1} + \\cdots + a_1 x + a_0 = 0\\) must be a factor of the constant term \\(a_0\\) divided by a factor of the leading coefficient \\(a_n\\). For our polynomial \\(4x^4 - 3x^2 + 7x - 3\\), the constant term is \\(-3\\) and the leading coefficient is \\(4\\). The possible rational roots are \\(\\pm 1, \\pm 3, \\pm \\frac{1}{2}, \\pm \\frac{3}{2}, \\pm \\frac{1}{4}, \\pm \\frac{3}{4}\\).\n\nWe test these possible roots by substitution:\n\n1. Testing \\(x = 1\\):\n   \\[\n   4(1)^4 - 3(1)^2 + 7(1) - 3 = 4 - 3 + 7 - 3 = 5 \\neq 0\n   \\]\n   So, \\(x = 1\\) is not a root.\n\n2. Testing \\(x = -1\\):\n   \\[\n   4(-1)^4 - 3(-1)^2 + 7(-1) - 3 = 4 - 3 - 7 - 3 = -9 \\neq 0\n   \\]\n   So, \\(x = -1\\) is not a root.\n\n3. Testing \\(x = \\frac{1}{2}\\):\n   \\[\n   4\\left(\\frac{1}{2}\\right)^4 - 3\\left(\\frac{1}{2}\\right)^2 + 7\\left(\\frac{1}{2}\\right) - 3 = 4\\left(\\frac{1}{16}\\right) - 3\\left(\\frac{1}{4}\\right) + \\frac{7}{2} - 3 = \\frac{4}{16} - \\frac{12}{16} + \\frac{56}{16} - \\frac{48}{16} = \\frac{4 - 12 + 56 - 48}{16} = 0\n   \\]\n   So, \\(x = \\frac{1}{2}\\) is a root.\n\nSince \\(x = \\frac{1}{2}\\) is a root, we can factor \\(4x^4 - 3x^2 + 7x - 3\\) as \\((2x - 1)(2x^3 + x^2 - x + 3)\\). To find the sum of the real roots of the original polynomial, we need to determine if the cubic polynomial \\(2x^3 + x^2 - x + 3\\) has any real roots. We can use Descartes' Rule of Signs to analyze the cubic polynomial \\(2x^3 + x^2 - x + 3\\). The number of sign changes in the sequence of coefficients (2, 1, -1, 3) is 2, so there are either 2 or 0 positive real roots. For the polynomial with the variable changed to \\(-x\\), we get \\(-2x^3 - x^2 + x + 3\\), which has 1 sign change, so there is exactly 1 negative real root.\n\nSince the cubic polynomial \\(2x^3 + x^2 - x + 3\\) has exactly 1 real root (because it must have an odd number of sign changes and no more than 1 negative real root), the sum of the real roots of the original polynomial is the sum of the real root \\(\\frac{1}{2}\\) and the real root of the cubic polynomial. Since the cubic polynomial has exactly one real root and we are not given its specific value, we can conclude that the sum of the real roots of the original polynomial is:\n\n\\[\n\\frac{1}{2} + \\text{(sum of the real root of the cubic polynomial)}\n\\]\n\nSince we know there is exactly one real root for the cubic polynomial, the sum of the real roots is:\n\n\\[\n\\frac{1}{2} + \\text{(the real root of the cubic polynomial)}\n\\]\n\nAs the cubic polynomial has exactly one real root, which we do not need to find to determine the sum of the real roots of the original polynomial, we know the sum of the real roots of the polynomial is:\n\n\\[\n\\frac{1}{2}\n\\]\n\nThus, the sum of the real roots of the equation is:\n\n\\[\n\\boxed{\\frac{1}{2}}\n\\]"
ground_truth = "-1"

res = default_compute_score(data_source, solution_str=response_str, ground_truth=ground_truth)
print(res)
