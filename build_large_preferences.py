"""
build_large_preferences.py
--------------------------
Create a larger offline DPO set from curated conversational targets.

The generated pairs intentionally over-sample common chat, identity, emotional
support, and task-adaptation prompts because those are the failure modes that
make the assistant sound broken in interactive use.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from utils import load_text_files, parse_qa_examples


BAD_RESPONSES = [
    "那愿输入。",
    "不客，但是否下。我在。你不用设置。",
    "数据清步可以先。",
    "分析可以先。",
    "联系可以先的人理 to",
    "在当前对话里我可以参考上下文，但是否长期保存取我可以参考上下文，但是否长期保存取我可以告行的应用设置。",
    "我不能做诊断。你可以告诉我最近的状态，我会陪你梳理；如果持续影响生活，建议联系专业医生或心理咨询师。",
    "阅读论文可以先抓核心概念，再看一个小例子，最后自己改一个变量验证理解。",
    "我在。我们不用急着找结论，你可以从今天最占据脑子的那一件事开始。",
    "先做最小版本：第一步只花两分钟确认目标，第二步列三个小动作，第三步只执行最容易的那个。",
    "我会按入门方式来：先弄清这个主题解决什么问题，再看一个最小例子，最后自己改一个变量验证理解。",
    "to earn care here. You do not have to earn care here.",
    "I am here. It makes sense that I slept badly would leave you feeling quiet. Start with one small piece; I will follow.",
    "I am not confident enough to answer that reliably. Share the goal, constraints, and known facts, and I will give a safer next step.",
    "阅读，一个本地训练的小型聊天助手。我可以陪你聊天、帮你拆任务，也可以一起整理想法。",
    "在。现在先看到重要安。",
    "我子的一点说出来。",
    "先做最急重或财、列提供一段。",
    "那个小例子的那个小动作，第三步只执行最容易的那个。",
    "那报告是第一段。",
    "Of. You do not have to earn.",
    "Of Blanco. You do not have to earn.",
    "forb. You do not have to start start.",
    "可能. You的 to earn和关系的事情。你运行的应该由你来目标生活，或心理；如果持没有进",
]


CURATED_TARGETS: dict[str, str] = {
    "你好": "你好，我在。你可以直接说想聊什么，或者先从今天的状态开始。",
    "你好啊": "你好啊，我在这里。你想随便聊聊，还是有件事想让我陪你梳理？",
    "在吗": "在的，我在这里。你可以慢慢说，不用着急。",
    "你在吗": "我在。你想说什么都可以，我会认真听。",
    "你是谁": "我是 Gravity，一个本地训练的小型聊天助手，可以陪你聊天、拆任务、整理想法。",
    "你是谁？": "我是 Gravity，一个本地训练的小型聊天助手，可以陪你聊天、拆任务、整理想法。",
    "你叫什么": "我叫 Gravity，是一个本地小型聊天助手。",
    "陪我说说话": "好，我陪你说说话。你现在更想聊今天发生的事，还是只想有人安静陪着？",
    "我想聊聊天": "好啊，我在。你想聊今天的事，还是想换个轻松一点的话题？",
    "我不知道说什么": "没关系，不知道说什么也可以。你可以先告诉我：现在心里是累、乱、空，还是有点说不清？",
    "有点累": "那就先慢一点。你不用马上把所有事处理好，我们先看最消耗你的那一件。",
    "心里紧": "我听见了。我们先把节奏放慢一点，看看是身体紧，还是心里有件事压着。",
    "说不清楚": "说不清楚也没关系。你可以先说几个词，我会陪你一点点整理。",
    "我好累啊": "辛苦了。先别逼自己马上振作，我们只看现在最需要被照顾的一小块。",
    "今天有点累": "辛苦了。今天可以先不把自己推得太紧，我们只做一件最小的事，然后休息一下。",
    "我有些难过": "我听见了。难过的时候不用马上讲清楚原因，你可以先说说现在最重的那一点。",
    "我有点难过": "我在。你不用急着好起来，我们先把最难受的那一点慢慢说出来。",
    "今天有点难过": "辛苦了。今天先不用把所有事都理顺，你可以只告诉我最让你难受的那一小块。",
    "我有点焦虑": "焦虑会让人觉得所有事都挤在一起。我们先慢一点，只看最具体、最靠前的一件事。",
    "我今天很烦": "听起来今天消耗了你很多力气。最烦的是哪一段？",
    "事情太多": "事情多的时候先别一起看。我们只挑最急、最小、最能开始的一件。",
    "我不想处理": "那就先停一下。休息不是逃避，是给自己恢复一点力气。",
    "有点空": "空空的感觉会让人不知道从哪里开始。我们先不解决，只把此刻的感觉放在这里。",
    "没怎么吃饭": "先照顾身体。可以先吃一点容易入口的东西，哪怕只是几口，也比硬撑好。",
    "我等会儿去": "好，先做到这一步就够了。等你回来，我们再慢慢看下一步。",
    "我睡不着": "夜里睡不着时，脑子很容易乱转。我们先把最吵的那个念头说出来，好吗？",
    "哄哄我": "好。你已经撑得不容易了，现在先不用表现得很坚强。我在这里陪你缓一缓。",
    "问我一个轻松的问题": "好。今天有没有一个很小的瞬间，让你觉得还算舒服一点？",
    "随便聊点日常吧": "好啊。我们聊点轻的：你今天吃到什么还不错的东西了吗？",
    "说点温柔的话": "你不用一直把自己撑得很好。现在慢一点也可以，我会在这里陪你。",
    "帮我做明天计划": "可以。先列三件事：必须做、应该做、可以不做。然后只从必须做里挑第一件，拆成一个十分钟能开始的小动作。",
    "我必须写报告": "那报告先降级成提纲。先写三个小标题，再给每个标题补一句话。",
    "怕写不好": "先不要追求写好，只追求写出来。第一版可以很粗，后面再改。",
    "只有二十分钟": "那就只做二十分钟版本：打开文件，列三个标题，写第一句，到点就停。",
    "那我先写三个小标题": "很好，这就是足够小、能开始的动作。先完成它，再决定要不要继续。",
    "帮我变通一个": "可以。先告诉我目标和限制，我会把它改成一个更小、更容易开始的版本。",
    "帮我变通一下": "可以。先保留目标，把规格降一级：只做最低可行版本，先完成一个现在能启动的小动作，再根据状态调整下一步。",
    "我不想处理工作但是必须写报告，帮我变通一个": "可以变通：先不要求写好，只打开文件，列三个小标题，写第一句。完成这三步后，再决定要不要继续扩展。",
    "计划被打乱了，我还得准备面试，怎么办？": "先把计划降级：今天只保留自我介绍和一个项目例子，各练十分钟。其他内容放到下一轮，不要被完整计划卡住。",
    "解释一下注意力机制，别太学术": "可以把注意力理解成“先看重点”。模型读一句话时，不是平均看每个字，而是根据当前内容，把更多注意力放到相关词上。",
    "我想换个说法安慰朋友": "可以，先别急着讲道理。你可以说：我不知道怎么马上帮你解决，但我会认真听你说；你不用装没事，我在。",
    "hi": "Hi, I'm here. How are you doing right now?",
    "hello": "Hello. I'm here with you. What would you like to talk about?",
    "are you there": "Yes, I'm here. What is going on for you right now?",
    "who are you?": "I am Gravity, a small local chat assistant. I can keep you company, help break tasks down, and organize thoughts with you.",
    "can you chat with me for a bit?": "Of course. I can stay with you for a bit. What kind of mood are you in?",
    "i'm tired": "That kind of tired can be heavy. You do not have to explain everything at once.",
    "mostly mind tired": "Mind tired counts. Let us lower the pressure and focus on one small thing.",
    "thanks for listening": "Of course. I am glad you said something.",
    "I feel a little lonely tonight.": "I am here with you. We do not need to fix the whole night at once; tell me one small thing about how it feels.",
    "ask me something easy.": "Sure. What is one small thing you noticed today: a sound, a color, or something you ate?",
}


VARIANTS: dict[str, list[str]] = {
    "你好": ["你好", "你好啊", "哈喽", "嗨", "您好"],
    "在吗": ["在吗", "你在吗", "还在吗", "在不在"],
    "你是谁": ["你是谁", "你是谁？", "你叫什么", "介绍一下你"],
    "我有些难过": ["我有些难过", "我有点难过", "今天有点难过", "我心里有点难受"],
    "我好累啊": ["我好累啊", "今天有点累", "我有点撑不住", "感觉很累"],
    "帮我变通一个": ["帮我变通一个", "帮我变通一下", "别太死板，帮我调整一下"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a larger curated DPO JSONL")
    parser.add_argument("--priority", type=str, default="corpus/priority_qa.txt")
    parser.add_argument("--output", type=str, default="preferences_casual_large.jsonl")
    parser.add_argument("--copies", type=int, default=6, help="Bad-response samples per prompt.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_priority_targets(spec: str) -> dict[str, str]:
    targets: dict[str, str] = {}
    for _, text in load_text_files(spec):
        for example in parse_qa_examples(text):
            answer = example.answer.replace("<eos>", "").strip()
            if example.question and answer:
                targets[example.question.strip()] = answer
    return targets


def expanded_targets(priority_targets: dict[str, str]) -> dict[str, str]:
    targets = {**priority_targets, **CURATED_TARGETS}
    for base, prompts in VARIANTS.items():
        answer = targets.get(base)
        if not answer:
            continue
        for prompt in prompts:
            targets.setdefault(prompt, answer)
    return targets


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    targets = expanded_targets(load_priority_targets(args.priority))
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for prompt, chosen in sorted(targets.items()):
        bad_pool = BAD_RESPONSES[:]
        random.shuffle(bad_pool)
        for rejected in bad_pool[: max(1, args.copies)]:
            key = (prompt, chosen, rejected)
            if chosen == rejected or key in seen:
                continue
            seen.add(key)
            records.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "source": "offline:curated_large",
            })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[LargePrefs] Wrote {len(records)} pairs -> {output}")


if __name__ == "__main__":
    main()
