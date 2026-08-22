from pydantic import BaseModel, Field


class ProfileBase(BaseModel):
    age: int | None = Field(default=None, ge=0, le=100)
    education: str | None = None
    major: str | None = None
    experience_years: int | None = Field(default=None, ge=0, le=60)
    target_position: str | None = None
    target_city: str | None = None
    expected_salary: str | None = None
    skills: str | None = None
    projects: str | None = None
    self_evaluation: str | None = None


class ProfileUpdate(ProfileBase):
    pass


class ProfileRead(ProfileBase):
    id: int
    user_id: int

    model_config = {"from_attributes": True}

