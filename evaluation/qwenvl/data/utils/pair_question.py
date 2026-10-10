import re
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# 1) 你可以直接把模板列表放进来
# -----------------------------
TEMPLATES = [
    {
        "meta": {"scene_type": "single_candidate", "slots": ["instance"]},
        "question": [
            "Is this the {instance} you're looking for?",
            "Is this the target {instance}?",
            "Is this the correct {instance}?",
            "Is this the right {instance}?",
            "Is this the {instance} you need?",
            "Does this look like the {instance} you want?",
            "Is this the {instance} you meant?",
            "Can you confirm this is the {instance}?",
            "Is this the {instance} you asked for?",
            "Is this the {instance} we are searching for?",
            "Is this one the {instance}?",
        ],
    },
    {
        "meta": {"scene_type": "multi_all_unseen", "slots": ["instance", "choice_ordinal"]},
        "question": [
            "Which one is the {instance}?",
            "Do you see the {instance} among these?",
            "Is the {instance} in this set? If so, which one?",
            "Which option matches the {instance}?",
            "Can you point out the {instance} here?",
            "Which candidate is the {instance}?",
            "Which one should I choose for the {instance}?",
            "Is any of these the {instance}? Which one?",
            "Which one looks like the target {instance}?",
            "Which one corresponds to the {instance}?",
            "Among these, which is the {instance}?",
        ],
        "answer": {
            "choice": [
                "The {choice_ordinal} one.",
                "{choice_ordinal}.",
                "Pick the {choice_ordinal} one.",
                "It's the {choice_ordinal} one.",
                "Choose the {choice_ordinal}.",
                "Go to the {choice_ordinal} one.",
                "The {choice_ordinal} option.",
                "Select the {choice_ordinal}.",
                "The {choice_ordinal} candidate.",
                "I mean the {choice_ordinal} one.",
                "The {choice_ordinal} one is correct.",
                "The {choice_ordinal} is the {instance}.",
            ],
            "none": [
                "None of them.",
                "None of these.",
                "No, the {instance} isn't here.",
                "No, none of these match.",
                "Neither-it's not shown here.",
                "No, it's not among these options.",
                "No, none of the candidates fit.",
                "No match here.",
                "No, this set doesn't include the {instance}.",
                "None of these is the target.",
                "No, keep searching.",
                "No, all of them are wrong.",
            ],
        },
    },
    {
        "meta": {
            "scene_type": "multi_confirmed_partial_except",
            "slots": ["instance", "except_clause", "choice_ordinal"]
        },
        "question": {
            "single": [
                "I've ruled out some candidates. Is the {except_clause} one the target {instance}?",
                "Only the {except_clause} one is still possible in this view. Is it the {instance}?",
                "We can ignore the others. Is the {except_clause} one the correct {instance}?",
                "Given what's ruled out, is the {except_clause} one the {instance} we want?",
                "After eliminating the rest, is the {except_clause} one the target {instance}?",
                "Is the remaining candidate—the {except_clause} one—the {instance}?",
                "I narrowed it down to the {except_clause} one in this view. Is that the {instance}?",
                "Is the {except_clause} one the right {instance}, yes or no?",
                "Can you confirm the {except_clause} one is the {instance}?",
                "Final check: is the {except_clause} one the {instance}?",
                "Does the {except_clause} one match the {instance}?",
                "Is the {except_clause} one the target instance?",
            ],
            "double": [
                "I've ruled out some candidates. In the current view, among the {except_clause} ones, which is the {instance}?",
                "I've ruled out some candidates. In the current view, among the {except_clause}, which is the {instance}?",
                "Only the {except_clause} candidates remain uncertain in the current view. Which one is the target {instance}?",
                "I narrowed it down to the {except_clause} ones in the current view. Which one should I pick for {instance}?",
                "I narrowed it down to the {except_clause} in the current view. Which one should I pick for {instance}?",
                "In the current view, between the uncertain options ({except_clause}), which is the {instance}?",
                "In this view, the uncertain candidates are {except_clause}. Which one is the {instance}?",
                "In the current view, I'm down to {except_clause}. Which one matches the {instance}?",
                "In this view, from {except_clause}, which candidate is the {instance}?",
                "In the current view, which of the uncertain candidates ({except_clause}) is the {instance}?",
                "In this view, please choose the {instance} from the uncertain options: {except_clause}.",
                "In the current view, among {except_clause}, which one should I select for {instance}?",
                "In this view, we only have {except_clause} left. Which one is correct for {instance}?",
                "In the current view, pick the {instance} from {except_clause}.",
            ],
        },
        "answer": {
            "single": {
                "positive": [
                    "Yes.",
                    "That's it!",
                    "Correct.",
                    "Yes, this is the one.",
                    "Yes, that's the target.",
                    "Exactly.",
                    "Yep.",
                    "Yes-pick this one.",
                    "Confirmed.",
                    "This is it.",
                    "Yes, this matches.",
                ],
                "negative": [
                    "No.",
                    "Not this one.",
                    "No, that's not it.",
                    "No, keep looking.",
                    "No, wrong {instance}.",
                    "Nope.",
                    "Not the target.",
                    "No, this isn't the {instance}.",
                    "No, skip this one.",
                    "No, try another.",
                    "No, this doesn't match.",
                    "No, not this {instance}.",
                ],
            },
            "double": {
                "choice": [
                    "The {choice_ordinal} one.",
                    "The {choice_ordinal}.",
                    "Pick the {choice_ordinal} one.",
                    "It's the {choice_ordinal} one.",
                    "Choose the {choice_ordinal}.",
                    "Go to the {choice_ordinal} one.",
                    "The {choice_ordinal} option.",
                    "Select the {choice_ordinal}.",
                    "The {choice_ordinal} candidate.",
                    "I mean the {choice_ordinal} one.",
                    "The {choice_ordinal} one is correct.",
                    "The {choice_ordinal} is the {instance}.",
                ],
                "none": [
                    "None of them.",
                    "Neither.",
                    "Neither one.",
                    "No, neither is correct.",
                    "No, none of these matches.",
                    "No, both are wrong.",
                    "Neither candidate is the {instance}.",
                    "No match among the remaining options.",
                    "None of the remaining candidates is correct.",
                    "No, keep looking.",
                    "No, not these.",
                    "No, it's neither of the two.",
                ],
            },
        },
    },
]


# -----------------------------
# 2) ordinal 归一化
# -----------------------------
def ordinal_suffix(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def number_to_ordinal_word(n: int) -> list[str]:
    """
    返回某个数字的英文序数词写法列表。
    例如:
        21 -> ["twenty-first", "twenty first"]
        100 -> ["hundredth", "one hundredth"]
    """
    if not (1 <= n <= 100):
        raise ValueError("Only supports 1~100")

    base_1_19 = {
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
        5: "fifth",
        6: "sixth",
        7: "seventh",
        8: "eighth",
        9: "ninth",
        10: "tenth",
        11: "eleventh",
        12: "twelfth",
        13: "thirteenth",
        14: "fourteenth",
        15: "fifteenth",
        16: "sixteenth",
        17: "seventeenth",
        18: "eighteenth",
        19: "nineteenth",
    }

    tens_cardinal = {
        20: "twenty",
        30: "thirty",
        40: "forty",
        50: "fifty",
        60: "sixty",
        70: "seventy",
        80: "eighty",
        90: "ninety",
    }

    tens_ordinal = {
        20: "twentieth",
        30: "thirtieth",
        40: "fortieth",
        50: "fiftieth",
        60: "sixtieth",
        70: "seventieth",
        80: "eightieth",
        90: "ninetieth",
    }

    if n <= 19:
        return [base_1_19[n]]

    if n == 100:
        return ["hundredth", "one hundredth"]

    if n in tens_ordinal:
        return [tens_ordinal[n]]

    tens = (n // 10) * 10
    ones = n % 10
    word = f"{tens_cardinal[tens]}-{base_1_19[ones]}"
    return [word, word.replace("-", " ")]


def build_ordinal_resources(max_n: int = 100):
    ordinal_word_to_num = {}
    ordinal_variants = []

    for n in range(1, max_n + 1):
        # 数字形式: 1st, 2nd, 3rd ...
        num_form = f"{n}{ordinal_suffix(n)}"
        ordinal_word_to_num[num_form] = n
        ordinal_variants.append(num_form)

        # 单词形式
        for w in number_to_ordinal_word(n):
            ordinal_word_to_num[w] = n
            ordinal_variants.append(w)

    # 为了 regex 更稳，长的放前面，避免部分匹配抢跑
    ordinal_variants = sorted(set(ordinal_variants), key=len, reverse=True)
    escaped_variants = [re.escape(x) for x in ordinal_variants]

    ordinal_pattern = (
        r"(?:the\s+)?"
        r"(?:"
        + "|".join(escaped_variants) +
        r")"
    )

    return ordinal_word_to_num, ordinal_pattern


ORDINAL_WORD_TO_NUM, ORDINAL_PATTERN = build_ordinal_resources(100)

def normalize_ordinal(text: Optional[str]) -> Optional[int]:
    if text is None:
        return None
    s = text.strip().lower()
    s = re.sub(r"^the\s+", "", s)
    return ORDINAL_WORD_TO_NUM.get(s)


# -----------------------------
# 3) 文本预处理
# -----------------------------
def normalize_text(text: str) -> str:
    text = text.strip()
    text = text.replace("—", "-").replace("–", "-")
    text = re.sub(r"\s+", " ", text)
    return text


# -----------------------------
# 4) 模板 -> regex
# -----------------------------
PLACEHOLDER_PATTERNS = {
    "instance": r"(?P<instance>.+?)",
    "except_clause": r"(?P<except_clause>.+?)",
    "choice_ordinal": rf"(?P<choice_ordinal>{ORDINAL_PATTERN})",
}

PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")

def template_to_regex(template: str) -> re.Pattern:
    """
    把:
        "The {choice_ordinal} one."
    转成宽松匹配的 regex
    """
    template = normalize_text(template)
    parts = []
    last = 0

    for m in PLACEHOLDER_RE.finditer(template):
        start, end = m.span()
        slot_name = m.group(1)

        # 先处理普通文本
        literal = template[last:start]
        literal = re.escape(literal)
        literal = literal.replace(r"\ ", r"\s+")
        parts.append(literal)

        # 再处理槽位
        slot_pat = PLACEHOLDER_PATTERNS.get(slot_name, rf"(?P<{slot_name}>.+?)")
        parts.append(slot_pat)

        last = end

    # 最后剩余普通文本
    literal = template[last:]
    literal = re.escape(literal)
    literal = literal.replace(r"\ ", r"\s+")
    parts.append(literal)

    # 允许句末标点轻微浮动
    regex_str = "".join(parts)

    # 模板本身若以标点结束，允许输入时少一点“戏剧张力”
    regex_str = regex_str.rstrip()
    regex_str = rf"^\s*{regex_str}\s*[.!?]?\s*$"

    return re.compile(regex_str, re.IGNORECASE)


# -----------------------------
# 5) 展开所有模板
# -----------------------------
def flatten_templates(templates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    输出每个叶子模板:
    {
        "scene_type": "...",
        "path": "answer.choice",
        "template": "The {choice_ordinal} one."
    }
    """
    flattened = []

    def walk(node: Any, path: List[str], scene_type: str):
        if isinstance(node, list):
            for item in node:
                flattened.append({
                    "scene_type": scene_type,
                    "path": ".".join(path),
                    "template": item,
                    "regex": template_to_regex(item),
                })
        elif isinstance(node, dict):
            for k, v in node.items():
                if k == "meta":
                    continue
                walk(v, path + [k], scene_type)

    for block in templates:
        scene_type = block["meta"]["scene_type"]
        for k, v in block.items():
            if k == "meta":
                continue
            walk(v, [k], scene_type)

    return flattened


# -----------------------------
# 6) 匹配函数
# -----------------------------
def score_match(template: str, groups: Dict[str, Optional[str]]) -> int:
    """
    简单打分：
    - 模板越长越优先
    - 捕获到的非空槽位越多越优先
    """
    non_empty_groups = sum(1 for v in groups.values() if v)
    return len(template) + 20 * non_empty_groups


def match_utterance(
    utterance: str,
    templates = TEMPLATES
) -> Optional[Dict[str, Any]]:
    """
    返回:
    {
        'scene_type': ...,
        'template_path': ...,
        'matched_template': ...,
        'slots': {
            'instance': ...,
            'except_clause': ...,
            'choice_ordinal': ...,
            'choice_ordinal_normalized': ...
        }
    }
    """
    utterance = normalize_text(utterance)
    candidates = []

    for item in flatten_templates(templates):
        m = item["regex"].match(utterance)
        if not m:
            continue

        groups = m.groupdict()
        result = {
            "scene_type": item["scene_type"],
            "template_path": item["path"],
            "matched_template": item["template"],
            "slots": {
                "instance": groups.get("instance"),
                "except_clause": groups.get("except_clause"),
                "choice_ordinal": groups.get("choice_ordinal"),
                "choice_ordinal_normalized": normalize_ordinal(groups.get("choice_ordinal")),
            },
        }
        candidates.append((score_match(item["template"], groups), result))

    if not candidates:
        # import ipdb;ipdb.set_trace()
        return None

    # 取分数最高的
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


# -----------------------------
# 7) 示例
# -----------------------------
if __name__ == "__main__":
    tests = [
        "Which one looks like the target stool?",
        "The 2nd one.",
        "Pick the second one.",
        "None of them.",
        "In the current view, among 2nd and 5th ones, which is the chair?",
        "I've ruled out some candidates. Is the 3rd one the target lamp?",
        "The third is the chair.",
    ]

    for t in tests:
        import ipdb;ipdb.set_trace()
        res = match_utterance(t, TEMPLATES)
        print(f"\nInput: {t}")
        print(res)