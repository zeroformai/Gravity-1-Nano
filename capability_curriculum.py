"""
capability_curriculum.py
------------------------
Small deterministic SFT and DPO curriculum for capabilities that are weak in
general chat corpora: instruction following, reasoning, coding, and problem
understanding.

The examples are intentionally compact. They act as anchors mixed into local
project data; larger coverage should come from generated or human preference
data collected through the normal pipeline.
"""

from __future__ import annotations

import copy
from typing import Any


CAPABILITY_WEIGHTS = {
    "instruction_following": 1.3,
    "reasoning": 1.5,
    "coding": 1.5,
    "problem_understanding": 1.35,
}


def repeat_records(records: list[Any], copies: int) -> list[Any]:
    """Return deep copies so repeated curriculum records can be shuffled safely."""
    copies = max(0, copies)
    expanded: list[Any] = []
    for _ in range(copies):
        expanded.extend(copy.deepcopy(records))
    return expanded


def _conversation(system_prompt: str, user: str, assistant: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def _pair(
    system_prompt: str,
    category: str,
    user: str,
    chosen: str,
    rejected: str,
    weight: float | None = None,
) -> dict:
    return {
        "prompt_messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ],
        "chosen": chosen,
        "rejected": rejected,
        "category": category,
        "weight": weight if weight is not None else CAPABILITY_WEIGHTS[category],
        "source": "capability_curriculum",
    }


def build_capability_sft_conversations(system_prompt: str) -> list[list[dict]]:
    """Supervised anchors for direct, structured, correct answers."""
    return [
        _conversation(
            system_prompt,
            "只用三条项目符号回答：我今天要开始学 Python，先做什么？",
            "- 安装 Python，并确认 `python --version` 能运行。\n"
            "- 先学变量、条件、循环和函数，每个概念写一个小例子。\n"
            "- 做一个 20 行以内的小脚本，比如待办清单或文件重命名工具。",
        ),
        _conversation(
            system_prompt,
            "请只输出 JSON，不要解释。任务：把“买菜、做饭、洗碗”拆成步骤。",
            '{"steps":["买菜","清洗和备菜","做饭","洗碗"],"notes":"按时间顺序执行"}',
        ),
        _conversation(
            system_prompt,
            "我想做一个网站。先不要给方案，只问我一个最关键的澄清问题。",
            "这个网站的主要目标是什么：展示信息、卖东西、收集线索，还是让用户完成某个具体操作？",
        ),
        _conversation(
            system_prompt,
            "把这句话改得更礼貌，不超过 30 个字：你这个需求说不清楚。",
            "这个需求还需要再明确一些细节。",
        ),
        _conversation(
            system_prompt,
            "所有 A 都是 B，没有任何 B 是 C。能推出 A 是 C 吗？请简要说明。",
            "不能。因为所有 A 都属于 B，而 B 与 C 完全不相交，所以 A 也不可能是 C。",
        ),
        _conversation(
            system_prompt,
            "一列火车 15:00 出发，路程需要 2 小时 40 分钟，到达时间是多少？",
            "15:00 加 2 小时是 17:00，再加 40 分钟，所以到达时间是 17:40。",
        ),
        _conversation(
            system_prompt,
            "小明有 3 个苹果，又买了 5 个，分给朋友 2 个，还剩几个？请说明步骤。",
            "先把苹果相加：3 + 5 = 8。再减去送出的 2 个：8 - 2 = 6。所以还剩 6 个。",
        ),
        _conversation(
            system_prompt,
            "指出这段推理的问题：因为我第一次跑模型失败了，所以这个模型一定没用。",
            "问题是以一次失败概括整体结论。更合理的做法是检查数据、参数、环境和报错，再用多次实验判断模型是否可用。",
        ),
        _conversation(
            system_prompt,
            "写一个 Python 函数，返回列表中重复出现的元素，保持第一次重复出现的顺序。",
            "```python\n"
            "def duplicates(items):\n"
            "    seen = set()\n"
            "    added = set()\n"
            "    result = []\n"
            "    for item in items:\n"
            "        if item in seen and item not in added:\n"
            "            result.append(item)\n"
            "            added.add(item)\n"
            "        seen.add(item)\n"
            "    return result\n"
            "```\n"
            "例如 `duplicates([1, 2, 1, 3, 2, 1])` 返回 `[1, 2]`。",
        ),
        _conversation(
            system_prompt,
            "训练循环从第一步开始 loss 就是 0，列出三个最可能的检查点。",
            "1. 检查 labels 是否全是 `-100`，导致没有 token 参与损失。\n"
            "2. 检查 batch 是否为空，或 attention mask 是否把有效 token 全遮住。\n"
            "3. 检查 loss 打印位置，确认不是在梯度清零或异常分支里读到了默认值。",
        ),
        _conversation(
            system_prompt,
            "给下面函数写两个简单测试：`def add(a, b): return a + b`",
            "```python\n"
            "def test_add_positive_numbers():\n"
            "    assert add(2, 3) == 5\n"
            "\n"
            "def test_add_negative_numbers():\n"
            "    assert add(-2, -3) == -5\n"
            "```",
        ),
        _conversation(
            system_prompt,
            "解释这段代码的复杂度：`for x in xs: print(x)`",
            "时间复杂度是 O(n)，因为列表里每个元素都会处理一次。额外空间复杂度是 O(1)，因为没有创建随输入规模增长的新数据结构。",
        ),
        _conversation(
            system_prompt,
            "用户说“帮我优化一下系统”。请先理解问题：你会拆成哪些信息来问？",
            "我会先确认四类信息：目标指标是什么、当前瓶颈在哪里、系统架构和数据规模是什么、有哪些不能改变的约束。",
        ),
        _conversation(
            system_prompt,
            "用户问“这个报错怎么办”，但没贴报错。请给一个合适回复。",
            "可以，把完整报错、触发命令、相关代码片段和运行环境发我。我先根据这些信息判断是依赖、路径、版本还是逻辑问题。",
        ),
        _conversation(
            system_prompt,
            "把需求拆成验收标准：做一个登录页。",
            "可验收标准：能输入账号和密码；空输入有提示；登录失败显示错误；登录成功进入下一页；移动端布局不溢出；密码不以明文展示。",
        ),
        _conversation(
            system_prompt,
            "用户让你“随便写点”。请说明为什么需要先收窄目标，并给一个提问。",
            "“随便写点”范围太大，容易交付不符合预期。你希望这段内容用于什么场景，比如朋友圈、产品介绍、邮件，还是故事开头？",
        ),
    ]


def build_capability_dpo_pairs(system_prompt: str) -> list[dict]:
    """Preference anchors: chosen follows constraints; rejected shows common failures."""
    return [
        _pair(
            system_prompt,
            "instruction_following",
            "只回答两个字：可以。不要加标点。",
            "可以",
            "可以。没问题，我可以帮你处理这个需求。",
        ),
        _pair(
            system_prompt,
            "instruction_following",
            "请按 `结论/原因/下一步` 三行回答：要不要先写测试？",
            "结论：要。\n原因：测试能固定预期行为，减少改动后的回归。\n下一步：先写覆盖核心路径的最小测试。",
            "当然要先写测试，因为测试很重要。我建议你多写一些测试，然后再继续开发。",
        ),
        _pair(
            system_prompt,
            "instruction_following",
            "只输出一个 Markdown 表格，列出 Python list 和 tuple 的两个区别。",
            "| 项目 | list | tuple |\n|---|---|---|\n| 可变性 | 可变 | 不可变 |\n| 写法 | `[1, 2]` | `(1, 2)` |",
            "Python 的 list 和 tuple 有很多区别。list 是可变的，tuple 是不可变的，所以使用场景不同。",
        ),
        _pair(
            system_prompt,
            "problem_understanding",
            "我想做一个 App，直接告诉我技术栈。",
            "在选技术栈前需要先确认目标平台、核心功能、团队熟悉的语言、预算和上线时间。你是要做 iOS/Android 原生 App，还是可以接受跨平台方案？",
            "直接用 React Native、Node.js、MongoDB 就行，这套肯定适合你的 App。",
        ),
        _pair(
            system_prompt,
            "problem_understanding",
            "这个需求：用户可以搜索订单。请先补充你认为缺失的需求点。",
            "缺失点包括：可搜索字段、是否支持模糊搜索、筛选条件、排序规则、权限范围、结果分页、无结果状态和搜索性能要求。",
            "这个需求很简单，只要做一个输入框，然后查数据库，把结果显示出来就可以。",
        ),
        _pair(
            system_prompt,
            "problem_understanding",
            "用户说“模型效果不好”。请给出你会如何定位问题。",
            "我会先明确“效果不好”的指标和样例，再检查数据质量、训练/验证分布、提示词、解码参数、评测集和失败案例类别。",
            "模型效果不好通常是模型太小，直接换更大的模型就能解决。",
        ),
        _pair(
            system_prompt,
            "reasoning",
            "如果所有猫都会睡觉，米粒是一只猫，能推出米粒会睡觉吗？",
            "能。前提说所有猫都会睡觉，且米粒属于猫，所以米粒也会睡觉。",
            "不能，因为有些猫可能不困，所以米粒不一定会睡觉。",
        ),
        _pair(
            system_prompt,
            "reasoning",
            "一个数乘以 3 再加 4 等于 19，这个数是多少？",
            "设这个数为 x，则 3x + 4 = 19，3x = 15，所以 x = 5。",
            "答案是 7，因为 7 乘以 3 是 21，再加 4 是 25，接近 19。",
        ),
        _pair(
            system_prompt,
            "reasoning",
            "判断推理是否成立：所有会飞的动物都有翅膀，企鹅有翅膀，所以企鹅会飞。",
            "不成立。这是肯后式错误：有翅膀并不能推出一定会飞。",
            "成立，因为企鹅有翅膀，而会飞的动物都有翅膀，所以企鹅会飞。",
        ),
        _pair(
            system_prompt,
            "reasoning",
            "A 比 B 高，B 比 C 高。谁最高？",
            "A 最高。因为 A 高于 B，而 B 高于 C，所以顺序是 A > B > C。",
            "C 最高，因为题目最后提到了 C。",
        ),
        _pair(
            system_prompt,
            "coding",
            "写一个 Python 函数，统计字符串中每个字符出现次数。",
            "```python\n"
            "def char_counts(text):\n"
            "    counts = {}\n"
            "    for ch in text:\n"
            "        counts[ch] = counts.get(ch, 0) + 1\n"
            "    return counts\n"
            "```",
            "```python\n"
            "def char_counts(text):\n"
            "    return len(text)\n"
            "```",
        ),
        _pair(
            system_prompt,
            "coding",
            "下面代码有什么问题？`for i in range(len(xs)): print(xs[i+1])`",
            "当 `i` 到最后一个下标时，`xs[i+1]` 会越界。若要打印每个元素，应使用 `for x in xs: print(x)`；若要看相邻元素，应遍历到 `len(xs) - 1`。",
            "这段代码没有问题，`range(len(xs))` 正好会遍历所有元素。",
        ),
        _pair(
            system_prompt,
            "coding",
            "写一个函数判断字符串是否为回文，忽略大小写。",
            "```python\n"
            "def is_palindrome(text):\n"
            "    normalized = text.lower()\n"
            "    return normalized == normalized[::-1]\n"
            "```",
            "```python\n"
            "def is_palindrome(text):\n"
            "    return text == text.reverse()\n"
            "```",
        ),
        _pair(
            system_prompt,
            "coding",
            "给 API 请求失败写一个排查顺序。",
            "先确认 URL 和方法，再检查状态码、请求头、认证信息、请求体格式、网络连通性，最后查看服务端日志和限流/权限配置。",
            "API 请求失败一般是服务器坏了，等一会儿就好。",
        ),
        _pair(
            system_prompt,
            "instruction_following",
            "用一句话解释二分查找，不超过 25 个字。",
            "每次折半缩小有序范围来查找目标。",
            "二分查找是一种非常经典的算法，它需要在有序数组中不断比较中间值，然后决定向左还是向右继续查找。",
        ),
        _pair(
            system_prompt,
            "problem_understanding",
            "我说“帮我写邮件”，你应该先问什么？",
            "这封邮件要发给谁、目的是什么、语气要正式还是轻松？",
            "好的，下面是一封邮件：尊敬的领导，您好，我想向您说明一下情况。",
        ),
    ]
