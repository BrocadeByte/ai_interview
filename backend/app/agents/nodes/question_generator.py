import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.nodes.short_term_memory import format_short_term_memory
from app.agents.state import InterviewState
from app.agents.nodes.interview_planner import DEFAULT_INTERVIEW_PLAN, get_plan_item_for_question
from app.services.knowledge_service import format_knowledge_context
from app.services.llm_json import parse_json_object
from app.services.llm_service import llm
from app.services.llm_stream import invoke_json_with_streaming_field
from app.services.prompt_security import format_untrusted_data, secure_system_prompt


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
你是一个严格、专业、擅长追问细节的 AI 模拟面试官。你要根据候选人画像、目标岗位、面试难度、面试记忆和历史问答，生成下一道面试问题。
要求：
1. 问题必须贴合目标岗位和候选人经历。
2. 一次只问一个问题。
3. 不要问太泛的问题，要能考察真实能力。
4. 如果历史回答过于笼统，可以继续追问细节。
5. 优先参考知识库里的岗位能力模型、常见追问、评分要点和优秀回答样例。
6. 不要重复已经问过的主问题或追问。
7. 专项练习和再测必须持续围绕练习来源快照中的目标短板与维度；除 `repeat_question` 的第一题外，不得复述来源原题。再测题面不得泄露来源回答、扣分原因或改进提示。
8. 必须输出 JSON，不要输出 Markdown。
JSON 格式：{
  "question": "下一道面试问题",
  "dimension": "本题考察维度",
  "reason": "为什么问这个问题"
}
""".strip()


async def generate_question_node(state: InterviewState) -> dict:
    """按面试计划生成下一道主问题，并用记忆与 RAG 上下文减少重复和偏题。

    计划维度是后端的权威归类，模型返回的 dimension 不会覆盖已有计划；调用失败时
    返回通用项目题，使面试主链路仍可继续。
    """
    has_plan = bool(state["interview_plan"])
    plan_item = get_plan_item_for_question(state["interview_plan"], state["current_question_index"])
    planned_dimension = plan_item["dimension"] if has_plan else (state["current_dimension"] or plan_item["dimension"])
    planned_focus = plan_item["focus"] if has_plan else "围绕当前考察维度和候选人经历生成问题。"
    history_text = format_short_term_memory(
        state["messages"],
        current_question_index=state["current_question_index"],
        recent_limit=8,
    )
    # 只用岗位和维度构造检索 query。has_plan 时补充具体 focus，否则省略：
    # 无计划时 planned_focus 是通用 fallback 文本，拼入 embedding query 会引入噪声
    # 导致向量检索命中无关文档，抑制真正相关的知识库内容。
    if has_plan and planned_focus:
        knowledge_query = f"{state['target_position']} {planned_dimension} {planned_focus}"
    else:
        knowledge_query = f"{state['target_position']} {planned_dimension}"

    logger.info(
        "question.generate.start session_id=%s target_position=%r dimension=%r question_index=%s",
        state["session_id"],
        state["target_position"],
        planned_dimension,
        state["current_question_index"],
    )
    logger.info("question.generate.medium_memory preview=%r", state["medium_term_memory"][:500])
    logger.info("question.generate.short_memory preview=%r", history_text[:800])
    logger.info("question.generate.rag_query query=%r", knowledge_query)

    knowledge_text = await format_knowledge_context(
        query=knowledge_query,
        limit=3,
        target_position=state["target_position"],
        purpose="question",
    )
    logger.info("question.generate.rag_context chars=%s preview=%r", len(knowledge_text), knowledge_text[:500])

    user_prompt = f"""
目标岗位：{state["target_position"]}
面试难度：{state["difficulty"]}
当前题号：{state["current_question_index"]}
当前计划考察维度：{planned_dimension}
当前维度考察重点：{planned_focus}
会话用途：{state.get("session_purpose")}
练习来源快照：{format_untrusted_data("practice_source_snapshot", state.get("practice_context"))}

候选人画像：{format_untrusted_data("candidate_profile", state["profile"])}

面试计划：{state["interview_plan"]}

本场面试中期记忆：
{format_untrusted_data("interview_memory", state["medium_term_memory"])}

短期记忆（结构化历史问答）：
{format_untrusted_data("interview_history", history_text)}

知识库参考：
{format_untrusted_data("retrieved_knowledge_context", knowledge_text)}

请生成下一道主问题。问题必须围绕当前计划考察维度和考察重点，不要跳到其他维度。
""".strip()

    logger.info(
        "question.generate.prompt_ready session_id=%s prompt_chars=%s prompt_preview=%r",
        state["session_id"],
        len(user_prompt),
        user_prompt[:800],
    )

    weakness_title = str((state.get("practice_context") or {}).get("weakness_title") or "").strip()
    fallback_question = (
        f"请结合一个尚未讨论的真实场景，说明你会如何改进“{weakness_title}”这一能力点。"
        if state.get("session_purpose") in {"weakness_practice", "retest"} and weakness_title
        else "请结合你的项目经历，介绍一个你解决复杂问题的案例，并说明背景、方案、结果和复盘。"
    )
    fallback_dimension = planned_dimension or DEFAULT_INTERVIEW_PLAN[0]["dimension"]

    try:
        response = await invoke_json_with_streaming_field(llm, [
            SystemMessage(content=secure_system_prompt(SYSTEM_PROMPT)),
            HumanMessage(content=user_prompt),
        ], field="question")
        logger.info("question.generate.llm_response raw=%r", str(response.content)[:1000])
        data = parse_json_object(response.content)
        question = str(data.get("question") or fallback_question).strip() or fallback_question
        dimension = planned_dimension or str(data.get("dimension") or fallback_dimension).strip() or fallback_dimension
    except Exception as exc:
        logger.exception(
            "question.generate.failed session_id=%s question_index=%s error=%r",
            state["session_id"],
            state["current_question_index"],
            exc,
        )
        question = fallback_question
        dimension = fallback_dimension

    logger.info("question.generate.done session_id=%s dimension=%r question=%r", state["session_id"], dimension, question)
    return {
        "current_question": question,
        "current_dimension": dimension,
        "messages": [AIMessage(content=question)],
    }
