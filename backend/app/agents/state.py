import operator
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


# 面试消息角色：assistant 表示 AI 面试官，user 表示求职者。
MessageRole = Literal["assistant", "user", "system"]

# 面试难度，和接口层 InterviewCreate 里的 difficulty 保持一致。
InterviewDifficulty = Literal["easy", "medium", "hard"]

# 面试会话状态，和 interview_sessions.status 字段保持一致。
InterviewStatus = Literal["active", "finished"]

InterviewAction = Literal["start", "answer", "finish"]


class InterviewMessageState(TypedDict):
    # 消息角色，用来区分这条消息是谁说的。
    role: MessageRole
    # 消息正文，保存问题、回答、系统提示等文本。
    content: str
    # 当前消息对应的考察维度，例如技术深度、项目经验、沟通协作。
    dimension: NotRequired[str]
    # 这条消息对应的主问题序号，便于后续统计每题评分。
    question_index: NotRequired[int]
    # 是否为追问消息，用来区分主问题和追问。
    is_followup: NotRequired[bool]


class InterviewPlanItem(TypedDict):
    # 考察维度名称，例如专业能力、项目经验、问题分析能力。
    dimension: str
    # 该维度计划提问数量。
    question_count: int
    # 该维度在总评分中的权重，建议用 0 到 1 的小数。
    weight: float
    # 该维度的提问重点。
    focus: str


class InterviewScoreState(TypedDict):
    # 被评分的题目序号。
    question_index: int
    # 被评分的问题。
    question: str
    # 用户对该问题的回答。
    answer: str
    # 当前题目所属考察维度。
    dimension: str
    # 总分，建议使用 0 到 100。
    score: int
    # 分维度评分，例如专业准确性、表达清晰度、岗位匹配度。
    sub_scores: dict[str, int]
    # 给用户看的评分理由。
    reason: str
    # 本题暴露出的不足。
    weaknesses: list[str]
    # 针对本题的回答优化建议。
    suggestions: list[str]
    # 是否使用了系统兜底评分。
    is_fallback: NotRequired[bool]
    # 兜底原因，正常模型评分为空。
    fallback_reason: NotRequired[str | None]
    # 本轮评分实际使用的知识检索快照。
    citations: NotRequired[list[dict[str, Any]]]


class FollowupDecisionState(TypedDict):
    # 是否需要继续追问。
    needs_followup: bool
    # 做出追问或不追问判断的原因。
    reason: str
    # 如果需要追问，这里保存追问问题。
    followup_question: NotRequired[str]


class InterviewState(TypedDict):
    action: InterviewAction
    # 当前登录用户 ID，对应 users.id。
    user_id: int
    # 当前面试会话 ID，对应 interview_sessions.id。
    session_id: int
    # 目标岗位，例如 Python 后端工程师、前端开发工程师。
    target_position: str
    # 当前面试难度。
    difficulty: InterviewDifficulty
    # 当前面试状态。
    status: InterviewStatus
    # 用户求职画像，来自 user_profiles 表。
    profile: dict[str, Any]
    # 简历结构化信息；第一版没有简历模块时可以为空。
    resume: dict[str, Any] | None
    # 目标岗位信息；后续接 JD 或岗位知识库时可以放更完整的岗位要求。
    target_job: dict[str, Any]
    # 面试计划，通常由 interview_planner 节点生成。
    interview_plan: list[InterviewPlanItem]
    # 当前正在考察的维度。
    current_dimension: str
    # 当前面试官问题。
    current_question: str
    # 当前主问题序号，从 1 开始。
    current_question_index: int
    # 完整上下文消息，用于问题生成、追问判断、报告生成；add_messages 会让 LangGraph 自动追加新消息。
    messages: Annotated[list[AnyMessage], add_messages]
    medium_term_memory: str
    # 每轮回答的评分结果；operator.add 会让 LangGraph 自动追加新的评分记录。
    scores: Annotated[list[InterviewScoreState], operator.add]
    # 整场面试累计发现的短板；operator.add 会让 LangGraph 自动追加新的短板。
    weaknesses: Annotated[list[str], operator.add]
    # 当前主问题下已经追问的次数。
    follow_up_count: int
    # 每个主问题允许的最大追问次数。
    max_follow_up_count: int
    # 最近一次追问判断结果。
    followup_decision: FollowupDecisionState | None
    # 最终报告；面试未结束时可以为空。
    final_report: dict[str, Any] | None


def create_initial_state(
    *,
    user_id: int,
    session_id: int,
    target_position: str,
    difficulty: InterviewDifficulty,
    profile: dict[str, Any],
) -> InterviewState:
    # 创建一份干净的初始状态，供创建面试会话或启动 LangGraph 时使用。
    return {
        "action": "start",
        "user_id": user_id,
        "session_id": session_id,
        "target_position": target_position,
        "difficulty": difficulty,
        "status": "active",
        "profile": profile,
        "resume": None,
        "target_job": {"position": target_position},
        "interview_plan": [],
        "current_dimension": "",
        "current_question": "",
        "current_question_index": 1,
        "messages": [],
        "medium_term_memory": "暂无本场面试摘要。",
        "scores": [],
        "weaknesses": [],
        "follow_up_count": 0,
        "max_follow_up_count": 2,
        "followup_decision": None,
        "final_report": None,
    }
