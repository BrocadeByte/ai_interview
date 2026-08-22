import re
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator, model_validator


NonEmptyText = Annotated[str, Field(min_length=1)]
ScoreValue = Annotated[StrictInt, Field(ge=0, le=100)]
MeaningfulTextList = Annotated[list[str], Field(min_length=1)]


REPORT_PLACEHOLDERS = {
    "无",
    "无内容",
    "暂无",
    "暂无内容",
    "没有",
    "没有内容",
    "不适用",
    "未知",
    "none",
    "null",
    "n/a",
}


def is_meaningful_report_text(value: str) -> bool:
    normalized = re.sub(r"[\s。！!，,；;：:]+", "", value).casefold()
    return bool(normalized) and normalized not in REPORT_PLACEHOLDERS


class StrictLlmOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InterviewPlanItemOutput(StrictLlmOutput):
    dimension: Annotated[str, Field(min_length=1, max_length=160)]
    question_count: Annotated[StrictInt, Field(ge=1, le=8)]
    weight: Annotated[float, Field(ge=0, le=1)]
    focus: NonEmptyText


class InterviewPlanOutput(StrictLlmOutput):
    question: NonEmptyText
    plan: Annotated[list[InterviewPlanItemOutput], Field(min_length=1, max_length=8)]


class VisibleQuestionOutput(StrictLlmOutput):
    question: Annotated[str, Field(min_length=1, max_length=2000)]

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must be non-empty")
        return cleaned


class AnswerPipelineOutput(StrictLlmOutput):
    needs_followup: StrictBool
    decision_reason: NonEmptyText
    question: str
    score: ScoreValue
    sub_scores: dict[str, ScoreValue]
    reason: NonEmptyText
    weaknesses: list[str]
    suggestions: list[str]

    @field_validator("sub_scores")
    @classmethod
    def validate_sub_scores(cls, value: dict[str, int]) -> dict[str, int]:
        required = {"专业准确性", "表达清晰度", "项目真实性", "岗位匹配度"}
        missing = required.difference(value)
        if missing:
            raise ValueError(f"sub_scores missing required keys: {sorted(missing)}")
        return value

    @field_validator("weaknesses", "suggestions")
    @classmethod
    def validate_list_items(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if not all(cleaned):
            raise ValueError("list items must be non-empty strings")
        return cleaned

    @model_validator(mode="after")
    def validate_question(self) -> "AnswerPipelineOutput":
        if self.needs_followup and not self.question.strip():
            raise ValueError("question is required when needs_followup is true")
        return self


class InterviewReportOutput(StrictLlmOutput):
    total_score: ScoreValue
    summary: NonEmptyText
    strengths: MeaningfulTextList
    weaknesses: MeaningfulTextList
    suggestions: MeaningfulTextList
    learning_path: MeaningfulTextList
    sample_answer: NonEmptyText

    @field_validator("summary", "sample_answer")
    @classmethod
    def validate_meaningful_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not is_meaningful_report_text(cleaned):
            raise ValueError("report text must contain meaningful content")
        return cleaned

    @field_validator("strengths", "weaknesses", "suggestions", "learning_path")
    @classmethod
    def validate_meaningful_list(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if not all(is_meaningful_report_text(item) for item in cleaned):
            raise ValueError("report list items must contain meaningful content")
        return cleaned


class InterviewMemoryOutput(StrictLlmOutput):
    summary: NonEmptyText
    covered_topics: list[str]
    strengths: list[str]
    weaknesses: list[str]
    next_focus: list[str]
