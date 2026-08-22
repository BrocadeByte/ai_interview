from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.nodes.short_term_memory import format_short_term_memory
from app.agents.nodes.interview_planner import get_plan_item_for_question
from app.agents.state import InterviewState
from app.services.citation_service import normalize_citations
from app.services.knowledge_service import format_knowledge_context, unpack_knowledge_context
from app.services.llm_json import as_dict, as_list, as_str, clamp_int, parse_json_object_with_repair
from app.services.llm_service import llm
from app.services.prompt_security import format_untrusted_data, secure_system_prompt


SYSTEM_PROMPT = """
你是一个严格、专业的 AI 面试评分官。你要根据面试问题、候选人回答、目标岗位、候选人画像、面试记忆、历史问答和知识库评分标准，对本轮回答进行评分。
评分要求：
1. 总分 score 使用 0 到 100 的整数。
2. sub_scores 至少包含：专业准确性、表达清晰度、项目真实性、岗位匹配度。
3. reason 要说明为什么这样打分。
4. weaknesses 写本题暴露的问题。
5. suggestions 写可执行的改进建议。
6. 必须输出 JSON，不要输出 Markdown。
JSON 格式：{
  "score": 80,
  "sub_scores": {
    "专业准确性": 80,
    "表达清晰度": 75,
    "项目真实性": 85,
    "岗位匹配度": 78
  },
  "reason": "评分理由",
  "weaknesses": ["不足1", "不足2"],
  "suggestions": ["建议1", "建议2"]
}
""".strip()


EXPECTED_SCHEMA = """
{
  "score": 80,
  "sub_scores": {"专业准确性": 80, "表达清晰度": 75, "项目真实性": 85, "岗位匹配度": 78},
  "reason": "评分理由",
  "weaknesses": ["不足1", "不足2"],
  "suggestions": ["建议1", "建议2"]
}
""".strip()


def _get_last_user_answer(state: InterviewState) -> str:
    for message in reversed(state["messages"]):
        if message.type == "human":
            return str(message.content)
    return ""


async def evaluate_answer_node(state: InterviewState) -> dict:
    """融合长短期记忆与 RAG 标准评价最新回答，返回可追加的评分和短板状态。

    模型输出无法修复时使用显式兜底分，保证状态图可继续运行，同时在评分理由中
    标记结果需要人工复核。
    """
    answer = _get_last_user_answer(state)
    plan_item = get_plan_item_for_question(state["interview_plan"], state["current_question_index"])
    current_dimension = str(state.get("current_dimension") or "").strip() or plan_item["dimension"]
    history_text = format_short_term_memory(
        state["messages"],
        current_question_index=state["current_question_index"],
        recent_limit=8,
    )
    knowledge_result = await format_knowledge_context(
        query=f"{state['target_position']} {current_dimension} {state['current_question']} {answer}",
        limit=3,
        target_position=state["target_position"],
        purpose="scoring",
        with_citations=True,
    )
    knowledge_text, raw_citations = unpack_knowledge_context(knowledge_result)
    citations = [
        citation.model_dump(mode="json")
        for citation in normalize_citations(
            raw_citations,
            question_index=state["current_question_index"],
        )
    ]

    user_prompt = f"""
目标岗位：{state["target_position"]}
面试难度：{state["difficulty"]}
当前题号：{state["current_question_index"]}
考察维度：{current_dimension}
计划维度：{plan_item["dimension"]}
计划考察重点：{plan_item["focus"]}

候选人画像：{format_untrusted_data("candidate_profile", state["profile"])}

当前面试问题：{state["current_question"]}

候选人本轮回答：{format_untrusted_data("candidate_answer", answer)}

本场面试中期记忆：
{format_untrusted_data("interview_memory", state["medium_term_memory"])}

短期记忆（结构化历史问答）：
{format_untrusted_data("interview_history", history_text)}

知识库参考：
{format_untrusted_data("retrieved_knowledge_context", knowledge_text)}

请对候选人本轮回答进行评分。
""".strip()

    response = await llm.ainvoke([
        SystemMessage(content=secure_system_prompt(SYSTEM_PROMPT)),
        HumanMessage(content=user_prompt),
    ])

    try:
        data = await parse_json_object_with_repair(
            response.content,
            llm=llm,
            expected_schema=EXPECTED_SCHEMA,
            max_retries=1,
        )
    except Exception:
        data = {
            "score": 60,
            "sub_scores": {},
            "reason": "评分模型输出格式异常，本轮使用系统兜底评分，建议人工复核。",
            "weaknesses": ["评分模型输出格式异常，无法可靠提取本轮短板。"],
            "suggestions": ["请稍后重试，或结合原始问答进行人工复核。"],
        }

    weaknesses = [as_str(item) for item in as_list(data.get("weaknesses"))]
    score_result = {
        "question_index": state["current_question_index"],
        "question": state["current_question"],
        "answer": answer,
        "dimension": current_dimension,
        "score": clamp_int(data.get("score"), default=60),
        "sub_scores": as_dict(data.get("sub_scores")),
        "reason": as_str(data.get("reason")),
        "weaknesses": weaknesses,
        "suggestions": [as_str(item) for item in as_list(data.get("suggestions"))],
        "citations": citations,
    }
    return {"scores": [score_result], "weaknesses": weaknesses}
