import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.graph_config import MAX_QUESTION_COUNT
from app.agents.state import InterviewPlanItem, InterviewState
from app.schemas.llm_outputs import InterviewPlanOutput
from app.services.knowledge_service import format_knowledge_context
from app.services.llm_json import as_list, as_str, parse_json_model_with_repair
from app.services.llm_service import llm
from app.services.llm_stream import invoke_json_with_streaming_field
from app.services.prompt_security import format_untrusted_data, secure_system_prompt


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
你是一个专业的 AI 面试规划器和面试官。你要根据候选人画像、本场绑定的 JD 快照、目标岗位、面试难度和知识库内容，生成第一道题并规划本场面试的考察维度。
要求：
1. question 必须贴合候选人经历、目标岗位、JD 硬性要求和第一个计划维度，一次只问一个具体问题。
2. 总主问题数必须等于 8。
3. 每个维度必须包含 dimension、question_count、weight、focus。
4. 维度要覆盖项目经验、岗位专业能力、系统设计或工程实践、问题排查与协作。
5. focus 要说明该维度具体考察什么。
6. 必须先输出 question 字段，再输出 plan 字段，以便流式推送首题。
7. 必须输出 JSON，不要输出 Markdown。
JSON 格式：{
  "question": "第一道面试问题",
  "plan": [
    {"dimension": "项目经验", "question_count": 2, "weight": 0.25, "focus": "围绕项目经历考察职责、难点、方案和结果"}
  ]
}
""".strip()


EXPECTED_SCHEMA = """
{
  "question": "第一道面试问题",
  "plan": [
    {"dimension": "项目经验", "question_count": 2, "weight": 0.25, "focus": "围绕项目经历考察职责、难点、方案和结果"}
  ]
}
""".strip()


DEFAULT_INTERVIEW_PLAN: list[InterviewPlanItem] = [
    {
        "dimension": "项目经验",
        "question_count": 2,
        "weight": 0.25,
        "focus": "围绕候选人的项目经历，考察职责、难点、方案和结果。",
    },
    {
        "dimension": "专业基础",
        "question_count": 2,
        "weight": 0.25,
        "focus": "考察目标岗位所需的基础知识、常用框架和工程实践。",
    },
    {
        "dimension": "系统设计",
        "question_count": 2,
        "weight": 0.25,
        "focus": "考察接口设计、数据存储、性能、安全和可扩展性。",
    },
    {
        "dimension": "问题排查与协作",
        "question_count": 2,
        "weight": 0.25,
        "focus": "考察线上问题定位、复盘能力、沟通协作和学习能力。",
    },
]


async def plan_interview_node(state: InterviewState) -> dict:
    """通过一次知识检索和一次模型调用生成面试计划与首题。"""
    knowledge_text = await format_knowledge_context(
        query=f"{state['target_position'].strip()} 面试计划 岗位能力模型 评分标准",
        limit=5,
        target_position=state["target_position"],
        purpose="planning",
    )
    user_prompt = f"""
目标岗位：{state["target_position"]}
面试难度：{state["difficulty"]}
候选人画像：{format_untrusted_data("candidate_profile", state["profile"])}
本场 JD 快照：{format_untrusted_data("job_description_snapshot", state["target_job"])}

知识库参考：
{format_untrusted_data("retrieved_knowledge_context", knowledge_text)}

请生成第一道面试问题，并规划本场面试的 8 个主问题考察维度。
必须先输出 question 字段，再输出 plan 字段。
""".strip()

    fallback_question = (
        "请结合你的项目经历，介绍一个你解决复杂问题的案例，"
        "并说明背景、方案、结果和复盘。"
    )

    try:
        response = await invoke_json_with_streaming_field(
            llm,
            [SystemMessage(content=secure_system_prompt(SYSTEM_PROMPT)), HumanMessage(content=user_prompt)],
            field="question",
        )
        output = await parse_json_model_with_repair(
            response.content,
            llm=llm,
            output_model=InterviewPlanOutput,
            max_retries=1,
        )
        plan = normalize_interview_plan([item.model_dump() for item in output.plan])
        question = output.question.strip() or fallback_question
    except Exception as exc:
        logger.exception("interview.plan.failed session_id=%s error=%r", state["session_id"], exc)
        plan = DEFAULT_INTERVIEW_PLAN
        question = fallback_question

    current_plan_item = get_plan_item_for_question(plan, state["current_question_index"])
    return {
        "interview_plan": plan,
        "current_dimension": current_plan_item["dimension"],
        "current_question": question,
        "messages": [AIMessage(content=question)],
    }


def normalize_interview_plan(raw_plan: object) -> list[InterviewPlanItem]:
    """清洗模型计划并将题量、权重调整为后续状态图可依赖的稳定结构。"""
    raw_items = [item for item in as_list(raw_plan) if isinstance(item, dict)]
    if not raw_items:
        return DEFAULT_INTERVIEW_PLAN

    plan: list[InterviewPlanItem] = []
    for raw in raw_items:
        dimension = as_str(raw.get("dimension")).strip()
        focus = as_str(raw.get("focus")).strip()
        if not dimension:
            continue
        try:
            question_count = int(raw.get("question_count") or 1)
        except (TypeError, ValueError):
            question_count = 1
        try:
            weight = float(raw.get("weight") or 0)
        except (TypeError, ValueError):
            weight = 0
        plan.append(
            {
                "dimension": dimension,
                "question_count": max(1, question_count),
                "weight": weight,
                "focus": focus or f"围绕{dimension}进行深入考察。",
            }
        )

    if not plan:
        return DEFAULT_INTERVIEW_PLAN

    return rebalance_plan(plan, MAX_QUESTION_COUNT)


def rebalance_plan(plan: list[InterviewPlanItem], total_questions: int) -> list[InterviewPlanItem]:
    """在不删除维度的前提下将总题数校准到上限，并统一维度权重。"""
    if total_questions <= 0:
        return []
    balanced = [dict(item) for item in plan[:total_questions]]
    current_total = sum(item["question_count"] for item in balanced)
    while current_total < total_questions:
        balanced[(current_total) % len(balanced)]["question_count"] += 1
        current_total += 1
    while current_total > total_questions and any(item["question_count"] > 1 for item in balanced):
        for item in reversed(balanced):
            if item["question_count"] > 1:
                item["question_count"] -= 1
                current_total -= 1
                break

    total_allocated = sum(item["question_count"] for item in balanced)
    for item in balanced:
        item["weight"] = item["question_count"] / total_allocated
    return balanced


def get_plan_item_for_question(plan: list[InterviewPlanItem], question_index: int) -> InterviewPlanItem:
    """按各维度题量的累计区间，定位指定主问题所属的计划项。"""
    if not plan:
        return DEFAULT_INTERVIEW_PLAN[0]
    cursor = 0
    for item in plan:
        cursor += int(item.get("question_count") or 0)
        if question_index <= cursor:
            return item
    return plan[-1]
