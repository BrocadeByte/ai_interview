from langchain_core.messages import AnyMessage


def format_short_term_memory(
    messages: list[AnyMessage],
    current_question_index: int,
    recent_limit: int = 8,
) -> str:
    if not messages:
        return "暂无历史问答。"

    selected_indexes = set(range(max(0, len(messages) - recent_limit), len(messages)))
    for index, message in enumerate(messages):
        if _metadata(message).get("question_index") == current_question_index:
            selected_indexes.add(index)

    lines: list[str] = []
    for index in sorted(selected_indexes):
        message = messages[index]
        meta = _metadata(message)
        question_index = int(meta.get("question_index") or 0)
        is_followup = bool(meta.get("is_followup") or False)
        followup_index = int(meta.get("followup_index") or 0)
        role = "AI" if message.type == "ai" else "用户"

        if message.type == "ai":
            if is_followup:
                label = f"第 {question_index} 题追问 {followup_index}"
            else:
                label = f"第 {question_index} 题主问题"
        else:
            if is_followup:
                label = f"第 {question_index} 题追问 {followup_index} 的回答"
            else:
                label = f"第 {question_index} 题回答"

        lines.append(f"{label}｜{role}：{message.content}")

    return "\n".join(lines) if lines else "暂无历史问答。"


def _metadata(message: AnyMessage) -> dict:
    return dict(getattr(message, "additional_kwargs", {}) or {})
