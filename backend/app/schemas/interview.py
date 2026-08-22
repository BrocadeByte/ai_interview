from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class InterviewCreate(BaseModel):
    target_position: str = Field(min_length=1, max_length=160)
    difficulty: str = Field(default="medium", pattern="^(easy|medium|hard)$")


class InterviewWarmup(BaseModel):
    target_position: str = Field(min_length=1, max_length=160)


class InterviewAnswer(BaseModel):
    answer: str = Field(min_length=1)
    request_id: UUID


class InterviewMessageRead(BaseModel):
    id: int
    role: str
    content: str
    question_index: int = 1
    dimension: str | None = None
    request_id: str | None = None
    is_followup: int = 0
    followup_index: int = 0
    created_at: datetime

    model_config = {"from_attributes": True}


class InterviewSessionRead(BaseModel):
    id: int
    target_position: str
    difficulty: str
    status: str
    current_question_index: int
    current_dimension: str | None = None
    current_plan_focus: str | None = None
    total_question_count: int = 8
    created_at: datetime
    updated_at: datetime
    messages: list[InterviewMessageRead] = []

    model_config = {"from_attributes": True}


class InterviewListItem(BaseModel):
    id: int
    target_position: str
    difficulty: str
    status: str
    current_question_index: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
