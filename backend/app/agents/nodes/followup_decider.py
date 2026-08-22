from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.nodes.interview_planner import get_plan_item_for_question
from app.agents.nodes.short_term_memory import format_short_term_memory
from app.agents.state import InterviewState
from app.services.llm_json import as_str, parse_json_object_with_repair
from app.services.llm_service import llm
from app.services.llm_stream import invoke_json_with_streaming_field
from app.services.prompt_security import format_untrusted_data, secure_system_prompt


SYSTEM_PROMPT = """
你是一个严格、专业的 AI 面试追问判断官。你要根据当前主问题链路、候选人回答、评分结果和面试记忆，判断是否需要继续追问。
判断规则：
1. 如果回答过于笼统、缺少细节、分数偏低，应该追问。
2. 如果回答已经完整、分数较高，可以不追问。
3. 如果已经达到当前问题的最大追问次数，必须不追问。
4. 追问必须基于候选人刚才回答中的缺口，不要换新主题。
5. 不要重复已经问过的追问。
6. 必须输出 JSON，不要输出 Markdown。
JSON 格式：{
  "needs_followup": true,
  "reason": "为什么需要或不需要追问",
  "followup_question": "如果需要追问，这里写追问问题；不需要追问则为空字符串"
}
""".strip()


EXPECTED_SCHEMA = """
{
  "needs_followup": true,
  "reason": "为什么需要或不需要追问",
  "followup_question": "如果需要追问，这里写追问问题；不需要追问则为空字符串"
}
""".strip()


def _get_last_user_answer(state: InterviewState) -> str:
    for message in reversed(state["messages"]):
        if message.type == "human":
            return str(message.content)
    return ""


async def decide_followup_node(state: InterviewState) -> dict:
    """依据最新评分和当前题链决定是否追问，并在需要时追加追问消息状态。

    最大追问次数由代码先行约束；模型异常或未给出问题文本时默认进入下一主问题，
    防止会话因不完整 JSON 停滞。
    """
    if not state["scores"]:
        return {"followup_decision": {"needs_followup": False, "reason": "没有评分结果，暂不追问。", "followup_question": ""}}

    if state["follow_up_count"] >= state["max_follow_up_count"]:
        return {"followup_decision": {"needs_followup": False, "reason": "已达到当前问题的最大追问次数。", "followup_question": ""}}

    answer = _get_last_user_answer(state)
    last_score = state["scores"][-1]
    plan_item = get_plan_item_for_question(state["interview_plan"], state["current_question_index"])
    history_text = format_short_term_memory(
        state["messages"],
        current_question_index=state["current_question_index"],
        recent_limit=8,
    )

    user_prompt = f"""
目标岗位：{state["target_position"]}
当前题号：{state["current_question_index"]}
当前计划维度：{plan_item["dimension"]}
当前维度考察重点：{plan_item["focus"]}
当前追问次数：{state["follow_up_count"]}
最大追问次数：{state["max_follow_up_count"]}

当前面试问题：{state["current_question"]}

候选人最新回答：{format_untrusted_data("candidate_answer", answer)}

评分结果：{last_score}

本场面试中期记忆：
{format_untrusted_data("interview_memory", state["medium_term_memory"])}

短期记忆（结构化历史问答）：
{format_untrusted_data("interview_history", history_text)}

请判断是否需要继续追问。如果追问，必须围绕当前计划维度和候选人刚才回答中的缺口，不要切换到其他维度。
""".strip()

    response = await invoke_json_with_streaming_field(llm, [
        SystemMessage(content=secure_system_prompt(SYSTEM_PROMPT)),
        HumanMessage(content=user_prompt),
    ], field="followup_question")

    try:
        data = await parse_json_object_with_repair(
            response.content,
            llm=llm,
            expected_schema=EXPECTED_SCHEMA,
            max_retries=1,
        )
    except Exception:
        data = {
            "needs_followup": False,
            "reason": "追问判断模型输出格式异常，默认不追问并进入下一题。",
            "followup_question": "",
        }

    needs_followup = bool(data.get("needs_followup"))
    followup_question = as_str(data.get("followup_question")).strip()
    if needs_followup and not followup_question:
        needs_followup = False

    update = {
        "followup_decision": {
            "needs_followup": needs_followup,
            "reason": as_str(data.get("reason")),
            "followup_question": followup_question,
        }
    }

    if needs_followup:
        update["messages"] = [
            AIMessage(
                content=followup_question,
                additional_kwargs={
                    "question_index": state["current_question_index"],
                    "is_followup": True,
                    "followup_index": state["follow_up_count"] + 1,
                },
            )
        ]
        update["current_question"] = followup_question
        update["follow_up_count"] = state["follow_up_count"] + 1

    return update
