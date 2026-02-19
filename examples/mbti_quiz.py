"""一个简单的 MBTI 性格测试小程序（命令行版）。

运行方式：
    python examples/mbti_quiz.py
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Question:
    text: str
    dimension: str
    option_a: str
    option_b: str


QUESTIONS: List[Question] = [
    Question("周末你更倾向于：", "EI", "和朋友聚会、社交", "独处放松、看书追剧"),
    Question("你在陌生环境中通常：", "EI", "主动开启对话", "先观察，熟悉后再交流"),
    Question("你更容易从哪里获得能量？", "EI", "外部互动和活动", "内在思考和独处"),
    Question("遇到新点子时你更常：", "SN", "关注未来可能性", "关注实际可行性"),
    Question("你做决定时更依赖：", "SN", "直觉与整体感觉", "事实与经验数据"),
    Question("阅读时你更喜欢：", "SN", "寓意和隐喻", "明确具体的信息"),
    Question("做决定时你更看重：", "TF", "逻辑与公平", "情感与关系"),
    Question("当朋友求助时你倾向于：", "TF", "先分析问题并给方案", "先共情安慰再建议"),
    Question("讨论问题时你更常：", "TF", "直接指出关键点", "考虑表达方式避免伤人"),
    Question("你对计划的态度是：", "JP", "提前安排，按计划推进", "保持灵活，随机应变"),
    Question("旅行前你通常会：", "JP", "详细制定行程", "大致方向，边走边看"),
    Question("截止日期临近时你更可能：", "JP", "提前完成并留余量", "在压力下冲刺完成"),
]

DIMENSION_MAP = {
    "EI": ("E", "I"),
    "SN": ("N", "S"),
    "TF": ("T", "F"),
    "JP": ("J", "P"),
}

TYPE_DESCRIPTIONS: Dict[str, str] = {
    "INTJ": "战略家：独立、理性、擅长长期规划。",
    "INTP": "逻辑学家：好奇、抽象思维强、热衷分析本质。",
    "ENTJ": "指挥官：目标导向、果断、擅长组织资源。",
    "ENTP": "辩论家：创意丰富、善于挑战常规和探索新可能。",
    "INFJ": "提倡者：理想主义、洞察力强、重视价值感。",
    "INFP": "调停者：真诚细腻、富同理心、重视内心契合。",
    "ENFJ": "主人公：善于鼓舞他人、重视关系与成长。",
    "ENFP": "竞选者：热情有想象力、喜欢探索新鲜事物。",
    "ISTJ": "物流师：可靠务实、重视秩序和责任。",
    "ISFJ": "守卫者：细心体贴、乐于支持他人。",
    "ESTJ": "总经理：执行力强、重视规则与效率。",
    "ESFJ": "执政官：热心合作、关注群体和谐。",
    "ISTP": "鉴赏家：冷静灵活、擅长动手和应急处理。",
    "ISFP": "探险家：温和敏感、注重体验与审美。",
    "ESTP": "企业家：行动力强、善于把握当下机会。",
    "ESFP": "表演者：活力外向、乐于分享快乐。",
}


def ask_question(question: Question, index: int) -> str:
    print(f"\nQ{index}. {question.text}")
    print(f"  A. {question.option_a}")
    print(f"  B. {question.option_b}")

    while True:
        answer = input("请选择 A 或 B：").strip().upper()
        if answer in {"A", "B"}:
            return answer
        print("输入无效，请输入 A 或 B。")


def calculate_type(scores: Dict[str, int]) -> str:
    result = []
    for dimension, (first_letter, second_letter) in DIMENSION_MAP.items():
        if scores[dimension] >= 0:
            result.append(first_letter)
        else:
            result.append(second_letter)
    return "".join(result)


def run_mbti_quiz() -> None:
    print("=" * 44)
    print("      欢迎使用 MBTI 性格测试（简化版）")
    print("=" * 44)
    print("说明：每题二选一，按第一反应作答即可。\n")

    scores = {"EI": 0, "SN": 0, "TF": 0, "JP": 0}

    for idx, question in enumerate(QUESTIONS, start=1):
        answer = ask_question(question, idx)
        scores[question.dimension] += 1 if answer == "A" else -1

    mbti_type = calculate_type(scores)
    description = TYPE_DESCRIPTIONS.get(mbti_type, "这是一个独特的类型组合，欢迎继续探索自我。")

    print("\n" + "-" * 44)
    print(f"你的测试结果是：{mbti_type}")
    print(f"类型解读：{description}")
    print("提示：该测试为简化娱乐版，不构成专业心理评估。")
    print("-" * 44)


if __name__ == "__main__":
    run_mbti_quiz()
