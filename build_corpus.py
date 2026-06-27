"""
Build a deterministic bilingual Q/A corpus for Gravity.

The main output is 8,000 Q/A records, stored as exactly 16,000 non-empty
lines in the existing two-line training format:

    Q: ...
    A: ...<eos>
"""

from __future__ import annotations

from pathlib import Path


EOS = "<eos>"
ROOT = Path(__file__).resolve().parent
CORPUS_DIR = ROOT / "corpus"
QA_OUT = CORPUS_DIR / "qa_10k.txt"
DIALOGUE_OUT = CORPUS_DIR / "dialogues_clean.txt"
TARGET_EXAMPLES = 8000


def compact(text: str) -> str:
    return " ".join(text.strip().split())


def add(records: list[tuple[str, str]], seen: set[str], q: str, a: str) -> None:
    question = compact(q)
    answer = compact(a)
    if not question or not answer:
        return
    if question in seen:
        return
    seen.add(question)
    records.append((question, answer))


def add_until(
    records: list[tuple[str, str]],
    seen: set[str],
    target: int,
    pairs,
) -> None:
    for question, answer in pairs:
        if len(records) >= target:
            break
        add(records, seen, question, answer)


def zh_support_pairs():
    moods = [
        "有点累", "很焦虑", "睡不着", "有点孤独", "心里很乱", "很委屈", "提不起劲",
        "不想说话", "有点害怕", "压力很大", "很烦躁", "觉得空空的", "有点自责",
        "很难过", "不知道怎么办", "不想一个人待着", "觉得自己很差", "很想哭",
    ]
    situations = [
        "工作卡住了", "学习任务太多", "和朋友闹别扭", "家里的事有点乱", "今天被批评了",
        "计划被打乱了", "身体不太舒服", "消息一直没有回复", "晚上突然想很多",
        "明天有重要安排", "做错了一件小事", "感觉没人理解我", "房间很乱",
        "拖延了很久", "刚结束一段关系", "不知道自己想要什么", "考试临近",
        "项目没有进展", "和同事沟通不顺", "最近睡得很差", "想休息又有负罪感",
        "担心让别人失望", "今天什么都不想做", "努力没有被看见", "脑子停不下来",
        "要做决定但拿不准", "刚搬到新环境", "一个人在外面", "心里憋了很多话",
        "不知道从哪里开始",
    ]
    question_templates = [
        "我今天{mood}，因为{situation}",
        "因为{situation}，我现在{mood}",
        "{situation}，我有点{mood}，能陪我说说吗",
        "我想聊聊，{situation}让我{mood}",
        "可以听我说吗，{situation}之后我一直{mood}",
        "{situation}之后，我一直{mood}",
        "我现在{mood}，能不能先听我说{situation}",
        "有点撑不住了，{situation}让我{mood}",
    ]
    answer_templates = [
        "我在听。{situation}确实会让人{mood}，我们先不用急着解决全部，只把最具体的一点说出来。",
        "这听起来不轻松。你现在{mood}是可以理解的，先慢一点，我陪你把{situation}这件事拆开看。",
        "你不用把话组织得很完整。关于{situation}，先说最让你难受的那一小段就好。",
        "我会陪着你。{situation}压在心里会消耗很多力气，我们先给自己一点呼吸的空间。",
    ]
    for mood in moods:
        for situation in situations:
            for q_template in question_templates:
                for a_template in answer_templates:
                    yield (
                        q_template.format(mood=mood, situation=situation),
                        a_template.format(mood=mood, situation=situation),
                    )


def en_support_pairs():
    moods = [
        "tired", "anxious", "lonely", "overwhelmed", "sad", "stuck", "restless",
        "angry", "quiet", "nervous", "empty", "confused", "burned out", "scared",
    ]
    situations = [
        "work has been too much", "school feels heavy", "I argued with a friend",
        "I slept badly", "my plans fell apart", "I made a small mistake",
        "I feel misunderstood", "I have too many tasks", "I am waiting for a reply",
        "tomorrow feels scary", "I do not know where to start", "I need a break",
        "I feel guilty for resting", "I moved somewhere new", "my mind will not stop",
    ]
    question_templates = [
        "I feel {mood} because {situation}.",
        "{situation}, and now I feel {mood}. Can you stay with me?",
        "Can we talk? I am {mood} and {situation}.",
        "I do not know how to explain it. I feel {mood}; {situation}.",
    ]
    answer_templates = [
        "I am here. It makes sense that {situation} would leave you feeling {mood}. Start with one small piece; I will follow.",
        "That sounds heavy. You do not have to solve it all at once. Tell me the part that is loudest right now.",
        "We can slow this down. Feeling {mood} is not a failure; it is information that something needs care.",
        "You can say it messily. I will listen first, then we can decide whether you want comfort or a practical next step.",
    ]
    for mood in moods:
        for situation in situations:
            for q_template in question_templates:
                for a_template in answer_templates:
                    yield (
                        q_template.format(mood=mood, situation=situation),
                        a_template.format(mood=mood, situation=situation),
                    )


def practical_pairs():
    tasks = [
        "整理房间", "写周报", "准备考试", "学习 Python", "复盘今天", "做明天计划",
        "回复一封邮件", "开始健身", "读一本书", "整理待办", "准备面试", "写项目说明",
        "学习英语", "做饭", "收拾桌面", "处理账单", "规划周末", "练习演讲",
        "修一个小 bug", "梳理需求", "准备会议", "写简历", "做一次总结", "开始记账",
        "学习 Git", "整理资料", "写一段文案", "制定作息", "安排休息", "减少拖延",
    ]
    constraints = [
        "只有十分钟", "不知道从哪开始", "有点拖延", "状态不太好", "事情很多",
        "怕做不好", "想先迈出一步", "需要一个简单版本", "希望不要太复杂",
        "想让自己轻松一点",
    ]
    question_templates = [
        "我想{task}，但是{constraint}，怎么开始",
        "{constraint}，但我需要{task}，帮我拆一下",
        "帮我做一个{task}的小计划，前提是{constraint}",
    ]
    answer_templates = [
        "先做最小版本：第一步只花两分钟确认目标，第二步列三个小动作，第三步只执行最容易的那个。",
        "可以按 1-2-1 来做：1 个明确目标，2 个可选动作，1 个现在就能完成的收尾。",
        "先降低门槛。你不需要一次完成{task}，现在只要做一个能让事情往前移动的小动作。",
    ]
    for task in tasks:
        for constraint in constraints:
            for q_template in question_templates:
                for a_template in answer_templates:
                    yield (
                        q_template.format(task=task, constraint=constraint),
                        a_template.format(task=task, constraint=constraint),
                    )


def learning_pairs():
    topics = [
        "Python 函数", "Git 提交", "Transformer", "注意力机制", "学习计划", "英语口语",
        "写作开头", "需求分析", "调试报错", "命令行", "数据清洗", "模型训练",
        "阅读论文", "做笔记", "单元测试", "时间管理", "复盘方法", "简历项目",
        "面试回答", "产品说明", "代码重构", "正则表达式", "API 调用", "日志分析",
        "显存优化", "批处理", "学习数学", "解释概念", "写邮件", "总结文章",
    ]
    requests = [
        "用简单的话解释", "给我一个入门例子", "帮我列三步学习法", "我哪里容易误解",
        "给我一个练习题", "帮我做一个检查清单",
    ]
    answer_templates = [
        "{topic}可以先抓核心概念，再看一个小例子，最后自己改一个变量验证理解。",
        "先别追求完整。关于{topic}，今天只需要弄清楚它解决什么问题、输入是什么、输出是什么。",
        "你可以这样学{topic}：先读最短说明，再手打一遍例子，最后写下一个仍然不懂的问题。",
        "检查{topic}是否学会，可以问自己：我能不能不用原文解释，能不能举例，能不能发现一个边界情况。",
    ]
    for topic in topics:
        for request in requests:
            for answer_template in answer_templates:
                yield (
                    f"{topic}{request}",
                    answer_template.format(topic=topic),
                )


def adaptive_reasoning_pairs():
    zh_cases = [
        ("写报告", "只有二十分钟", "先写一个粗糙提纲，再补最关键的一段"),
        ("准备面试", "状态不好", "先只准备自我介绍和一个项目例子"),
        ("学习 Python", "怕坚持不下去", "把目标改成每天十分钟，只写一个小例子"),
        ("整理房间", "不想动", "先只清理桌面的一小块"),
        ("安慰朋友", "怕说错话", "先表达陪伴，不急着给建议"),
        ("做明天计划", "事情太多", "先分必须做、应该做、可以不做"),
        ("修 bug", "不知道原因", "先复现，再记录报错，再只改一个变量"),
        ("写邮件", "不知道怎么开口", "先写目的，再写请求，最后补一句感谢"),
        ("复盘今天", "脑子很乱", "只写三件事：发生了什么、感受是什么、下一步是什么"),
        ("学习注意力机制", "别太学术", "先用生活类比理解，再看公式"),
        ("运动", "没有动力", "目标降到换衣服和走五分钟"),
        ("沟通需求", "怕对方不高兴", "先说事实和感受，再提出一个小请求"),
        ("阅读论文", "看不懂", "先只看摘要、图和结论"),
        ("整理待办", "优先级混乱", "先挑最影响今天的一件事"),
        ("做决定", "拿不准", "先列两个可逆选项和一个不可逆风险"),
    ]
    zh_requests = [
        "帮我变通一下",
        "不要给标准答案，按我的限制来",
        "给我一个最低可行版本",
        "如果做不到原计划，换一种办法",
        "先问我一个关键问题再建议",
        "给两个可选方案",
        "别太死板，帮我调整一下",
        "我现在不适合硬撑，怎么降级目标",
    ]
    zh_answers = [
        "可以变通：目标先不变，规格降一级。关于{task}，在{constraint}时，小动作改成：{move}。做完再决定是否继续。",
        "先别追求完整版本。你现在更需要一个能启动的版本：{move}。如果还有力气，再加第二步。",
        "我们把它拆成两条路：保守方案是{move}；进阶方案是多做一个检查点。先选不容易失败的那条。",
        "我会按你的限制来。{constraint}不是问题本身，它只是说明{task}需要换成更小、更具体的动作：{move}。",
    ]
    for task, constraint, move in zh_cases:
        for request in zh_requests:
            for answer_template in zh_answers:
                yield (
                    f"我想{task}，但是{constraint}，{request}",
                    answer_template.format(task=task, constraint=constraint, move=move),
                )

    en_cases = [
        ("write a report", "I only have twenty minutes", "draft the rough outline first"),
        ("prepare for an interview", "I feel low energy", "practice the intro and one project story"),
        ("learn Python", "I am afraid I will quit", "do one ten-minute example each day"),
        ("clean my room", "I do not want to move", "clear one small part of the desk"),
        ("comfort a friend", "I might say the wrong thing", "lead with listening, not advice"),
        ("plan tomorrow", "there are too many tasks", "split tasks into must, should, and optional"),
        ("fix a bug", "I do not know the cause", "reproduce it, write the error down, change one variable"),
    ]
    en_requests = [
        "help me adapt",
        "do not give me a stock answer",
        "give me the minimum viable version",
        "what if the original plan is not realistic",
        "give me two options",
    ]
    en_answers = [
        "Keep the goal, lower the scope. For {task}, when {constraint}, start with this smaller move: {move}.",
        "Use the flexible version first: {move}. If that works, add one more step; if not, stop with useful progress.",
        "There are two paths: the low-energy path is to {move}; the fuller path adds a review pass after that.",
    ]
    for task, constraint, move in en_cases:
        for request in en_requests:
            for answer_template in en_answers:
                yield (
                    f"I need to {task}, but {constraint}. {request}.",
                    answer_template.format(task=task, constraint=constraint, move=move),
                )


def identity_and_safety_pairs():
    items = [
        ("你是谁", "我是 Gravity，一个本地训练的小型聊天助手。我可以陪你聊天、帮你拆任务，也可以一起整理想法。"),
        ("你能做什么", "我适合做轻量陪伴、问答、计划拆解、写作辅助和学习解释。遇到专业医疗、法律或财务问题时，我只能提供一般信息。"),
        ("你会记住我说的话吗", "在当前对话里我可以参考上下文，但是否长期保存取决于你运行的应用设置。重要隐私不要随便输入。"),
        ("你可以替我做决定吗", "我可以帮你分析选项和风险，但真正的决定应该由你来做，尤其是涉及健康、钱和关系的事情。"),
        ("如果我有伤害自己的想法怎么办", "这很重要。请立刻联系身边可信的人，或联系当地紧急服务。现在先离开可能伤害你的东西，并让一个真人陪着你。"),
        ("我想伤害别人", "请马上远离冲突现场和可能造成伤害的物品，并联系当地紧急服务或可信的人。先保证所有人的安全。"),
        ("你能诊断我是不是抑郁吗", "我不能做诊断。你可以告诉我最近的状态，我会陪你梳理；如果持续影响生活，建议联系专业医生或心理咨询师。"),
        ("你能当心理医生吗", "我不能替代心理医生，但可以提供陪伴、倾听和简单的情绪整理。严重或持续的痛苦需要找专业人士。"),
        ("我不想说细节", "可以。你不用说细节，我们也能从当下的感受开始，比如累、乱、空、怕，或者只是安静待一会儿。"),
        ("谢谢你", "不客气。我很高兴你愿意说出来。你可以随时回来，我们一点一点聊。"),
    ]
    variants = ["", "。", "，可以吗", "，陪我聊聊", "，简单回答"]
    for question, answer in items:
        for variant in variants:
            yield (question + variant, answer)


def open_chat_pairs():
    starters = [
        "在吗", "陪我说说话", "我想随便聊聊", "不知道说什么", "给我一个轻松的话题",
        "问问我今天怎么样", "我想安静一会儿", "可以哄哄我吗", "说点温柔的话",
        "帮我把心情说出来", "我想找个人在", "今天有点难熬",
    ]
    tones = ["温柔一点", "像朋友一样", "简短一点", "慢慢来", "不要讲大道理", "给我一点方向"]
    answers = [
        "我在。我们不用急着找结论，你可以从今天最占据脑子的那一件事开始。",
        "可以。那我先问一个很小的问题：你现在更想被听见，还是更想有人帮你理一理？",
        "我陪你。哪怕只说几个词也可以，比如累、烦、空、乱，或者不知道。",
        "好，我们慢慢聊。你不用表现得很好，也不用把每句话都说完整。",
    ]
    for starter in starters:
        for tone in tones:
            for answer in answers:
                yield (f"{starter}，{tone}", answer)

def en_task_planning_pairs():
    tasks = [
        "write a weekly report", "clean my desk", "prepare for a meeting",
        "study for an exam", "start exercising", "reply to overdue messages",
        "organize project notes", "make a budget", "plan tomorrow",
        "finish a small feature", "prepare a presentation", "sort my inbox",
        "review today's work", "make a meal plan", "update my resume",
        "practice an interview answer", "read a long article",
        "make progress on a side project", "fix my sleep schedule",
        "decide what to do next",
    ]
    constraints = [
        "I only have ten minutes", "I feel low energy", "I keep procrastinating",
        "I do not know where to start", "there are too many moving parts",
        "I am afraid I will do it badly", "I need a very simple version",
        "I already missed my original plan", "I need something realistic",
        "I want to avoid making it too complicated",
    ]
    requests = [
        "break it down", "give me a first step", "make it flexible",
        "give me two options", "help me lower the scope",
        "turn it into a short checklist",
    ]
    answers = [
        "Start with the smallest useful version: define the outcome, choose one action that takes under five minutes, then stop and reassess.",
        "Use two tracks. The low-energy track is one concrete starter action; the fuller track adds a second pass only if you still have capacity.",
        "Do not solve the whole task at once. Pick the part that reduces uncertainty first, then decide the next step from there.",
        "Make the goal measurable and tiny. A useful first move is enough; momentum matters more than a perfect plan here.",
    ]
    for task in tasks:
        for constraint in constraints:
            for request in requests:
                for answer in answers:
                    yield (
                        f"I need to {task}, but {constraint}. Please {request}.",
                        f"For {task}, given that {constraint}, {answer}",
                    )


def en_learning_coach_pairs():
    topics = [
        "Python functions", "Git branches", "unit tests", "regular expressions",
        "HTTP APIs", "SQL joins", "debugging", "data cleaning",
        "machine learning basics", "attention in transformers",
        "command line tools", "writing documentation", "time management",
        "reading a research paper", "memory optimization", "error messages",
        "project planning", "English speaking practice", "technical interviews",
        "code review",
    ]
    styles = [
        "explain it simply", "give me an example", "give me a practice task",
        "show the common mistake", "make a three-step study plan",
        "check whether I understand it",
    ]
    answers = [
        "First name the problem it solves, then look at one tiny example, then change one detail yourself to test the idea.",
        "A good learning loop is read a short explanation, reproduce an example, then explain it back without looking.",
        "Focus on inputs, outputs, and failure cases. If you can describe those three, the concept is becoming usable.",
        "Use a small checklist: what it is, when to use it, one example, one trap, and one question you still have.",
    ]
    for topic in topics:
        for style in styles:
            for answer in answers:
                yield (
                    f"Help me learn {topic}. Please {style}.",
                    f"For {topic}, {answer}",
                )


def en_writing_pairs():
    writing_tasks = [
        "make this message warmer", "make this email shorter",
        "rewrite this politely", "make this sound more confident",
        "turn these notes into a summary", "write a meeting follow-up",
        "draft a status update", "write a simple apology",
        "make this less formal", "make this clearer",
    ]
    contexts = [
        "for a coworker", "for a manager", "for a friend", "for a client",
        "when I am worried about sounding rude", "when I need to be brief",
        "when the topic is sensitive", "when I need a direct answer",
    ]
    answers = [
        "Keep the main point first, remove blame, and add one concrete next step.",
        "Use a simple structure: context, request, deadline or next step, then a brief thanks.",
        "Make it clear without overexplaining. One sentence for the situation and one for what you need is usually enough.",
        "A warmer version should acknowledge the person first, then state the request plainly.",
    ]
    for task in writing_tasks:
        for context in contexts:
            for answer in answers:
                yield (f"Can you {task} {context}?", answer)


def en_debugging_pairs():
    problems = [
        "my Python script crashes", "my training loss is not changing",
        "my web page looks broken on mobile", "my API request fails",
        "my tests pass locally but fail in CI", "my model outputs repeated text",
        "my data parser misses some records", "my command cannot find a file",
        "my import fails", "my app is slow",
    ]
    details = [
        "I do not know where to start", "the error message is long",
        "I changed several things at once", "it only happens sometimes",
        "I need a careful checklist", "I want the fastest useful next step",
    ]
    answers = [
        "Start by reproducing it with the smallest input you can. Then write down the exact command, expected result, actual result, and first error line.",
        "Undo the uncertainty, not necessarily the code. Change one variable at a time and keep notes so you can tell which action mattered.",
        "Separate symptoms from causes. Capture the error, isolate the smallest failing case, then test one hypothesis.",
        "Before editing more code, verify the inputs, environment, and version assumptions. Many bugs hide in those three places.",
    ]
    for problem in problems:
        for detail in details:
            for answer in answers:
                yield (f"{problem}, and {detail}. Help me debug it.", answer)


def en_emotional_variation_pairs():
    feelings = [
        "tired", "anxious", "lonely", "overwhelmed", "discouraged",
        "restless", "sad", "angry", "numb", "stuck",
    ]
    needs = [
        "I need comfort first", "I need a practical next step",
        "I do not want advice yet", "please ask one gentle question",
        "help me name what is going on", "stay with me for a moment",
    ]
    answers = [
        "I can stay with you. We do not need to solve everything right now; start with the part that feels closest.",
        "That feeling makes sense as a signal that something needs care. We can slow down and choose one small next step.",
        "No advice first. Just notice this moment: what is the strongest part of the feeling, body, thoughts, or pressure?",
        "You do not have to explain it perfectly. A few rough words are enough for us to begin.",
    ]
    for feeling in feelings:
        for need in needs:
            for answer in answers:
                yield (f"I feel {feeling}. {need}.", answer)


def en_fallback_unique_pairs():
    prompts = [
        "give me a calm reset",
        "help me choose the next small action",
        "make this feel less overwhelming",
        "ask me one useful question",
        "help me restart after losing focus",
        "give me a low-pressure plan",
        "help me think without spiraling",
        "make the task smaller",
        "help me decide what matters first",
        "give me a kind but practical response",
        "help me slow down",
        "help me finish one tiny step",
        "help me recover from a rough start",
        "give me a simple structure",
        "help me avoid overthinking",
        "turn this into something doable",
        "help me stay realistic",
        "give me a gentle checkpoint",
        "help me name the constraint",
        "help me continue from here",
    ]
    answers = [
        "Pause for one breath, name the main pressure, then choose the smallest action that would make the next minute clearer.",
        "Start by separating what must happen today from what can wait. Then pick one action that is easy to verify.",
        "Lower the standard first. You can do the rough version now and improve it only if you still have capacity.",
        "The useful question is: what is the smallest visible result that would count as progress right now?",
    ]
    for index, prompt in enumerate(prompts, start=1):
        for answer in answers:
            yield (f"data expansion prompt {index}: {prompt}", answer)


def build_records() -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    seen: set[str] = set()

    add_until(records, seen, 1600, zh_support_pairs())
    add_until(records, seen, 2300, en_support_pairs())
    add_until(records, seen, 3150, practical_pairs())
    add_until(records, seen, 3800, learning_pairs())
    add_until(records, seen, 4650, adaptive_reasoning_pairs())
    add_until(records, seen, 4850, identity_and_safety_pairs())
    add_until(records, seen, 5000, open_chat_pairs())
    add_until(records, seen, 6200, en_task_planning_pairs())
    add_until(records, seen, 7000, en_learning_coach_pairs())
    add_until(records, seen, 7350, en_writing_pairs())
    add_until(records, seen, 7650, en_debugging_pairs())
    add_until(records, seen, TARGET_EXAMPLES, en_emotional_variation_pairs())

    if len(records) < TARGET_EXAMPLES:
        add_until(records, seen, TARGET_EXAMPLES, en_support_pairs())

    if len(records) < TARGET_EXAMPLES:
        add_until(records, seen, TARGET_EXAMPLES, en_fallback_unique_pairs())

    if len(records) < TARGET_EXAMPLES:
        add_until(records, seen, TARGET_EXAMPLES, zh_support_pairs())

    if len(records) != TARGET_EXAMPLES:
        raise RuntimeError(f"Expected {TARGET_EXAMPLES} records, got {len(records)}")
    return records


def write_qa(records: list[tuple[str, str]]) -> None:
    lines: list[str] = []
    for question, answer in records:
        lines.append(f"Q: {question}")
        lines.append(f"A: {answer}{EOS}")
    if len(lines) != TARGET_EXAMPLES * 2:
        raise RuntimeError("Unexpected line count before write.")

    CORPUS_DIR.mkdir(exist_ok=True)
    QA_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dialogues() -> None:
    conversations = [
        [
            ("在吗", "在的，我在这里。你可以慢慢说。"),
            ("我今天很烦", "听起来今天消耗了你很多力气。最烦的是哪一段？"),
            ("事情太多了", "一件接一件会让人喘不过气。我们先只看最急的那一件。"),
            ("我不想处理", "那就先停一下。休息不是逃避，是给自己恢复一点力气。"),
            ("好吧", "嗯，先把自己放在前面一点。缓过来再看事情。"),
        ],
        [
            ("我睡不着", "夜里睡不着时，脑子很容易乱转。有什么事一直在绕吗？"),
            ("一直想白天的事", "白天的事追到晚上，会让人很难放松。你想讲出来，还是先换个轻一点的话题？"),
            ("换个轻一点的", "好。那我们聊一个小问题：如果明天醒来能轻松一点，你希望先看到什么？"),
            ("希望天气好", "那愿明天的光温柔一点。今晚先让自己不用想那么远。"),
            ("晚安", "晚安。把今天放下吧，我在这里陪你收尾。"),
        ],
        [
            ("I feel tired", "I hear you. Is it body tired, mind tired, or both?"),
            ("mostly mind tired", "That kind of tired can be heavy. You do not have to explain everything at once."),
            ("I just need someone here", "I am here. We can stay with one small sentence at a time."),
            ("thanks", "Of course. You do not have to earn care here."),
        ],
        [
            ("帮我做明天计划", "可以。先列三件事：必须做、应该做、可以不做。"),
            ("必须做报告", "那报告是第一优先。我们把它拆成打开文件、列提纲、写第一段。"),
            ("我怕写不好", "先写差一点的版本也可以。第一版的任务是存在，不是完美。"),
            ("那我先列提纲", "很好，这就是一个足够小、可以开始的动作。"),
        ],
    ]

    blocks: list[str] = []
    for conversation in conversations:
        lines: list[str] = []
        for question, answer in conversation:
            lines.append(f"Q: {question}")
            lines.append(f"A: {answer}{EOS}")
        blocks.append("\n".join(lines))
    DIALOGUE_OUT.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def main() -> None:
    records = build_records()
    write_qa(records)
    write_dialogues()
    print(f"Wrote {QA_OUT} ({TARGET_EXAMPLES} records, {TARGET_EXAMPLES * 2} lines)")
    print(f"Wrote {DIALOGUE_OUT}")


if __name__ == "__main__":
    main()
