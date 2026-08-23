import re
from collections.abc import Mapping
from typing import Any

from app.schemas.job_description import ParsedJobDescription
from app.schemas.profile import AutoProfileDraft, ProfileFieldSource, ProfileUpdate
from app.schemas.resume import ParsedResume, ResumeProfilePatch


PROFILE_COMPLETENESS_WEIGHTS = {
    "target_position": 20,
    "skills": 20,
    "projects": 20,
    "experience_years": 10,
    "education": 10,
    "major": 5,
    "self_evaluation": 10,
    "target_city": 5,
}
PROFILE_FIELD_LABELS = {
    "target_position": "目标岗位",
    "skills": "技能栈",
    "projects": "项目经历",
    "experience_years": "工作/项目年限",
    "education": "学历",
    "major": "专业",
    "self_evaluation": "自我评价",
    "target_city": "目标城市",
}
LOW_COMPLETENESS_THRESHOLD = 70


def build_auto_profile_draft(
    *,
    existing_profile: Mapping[str, Any] | None,
    resume_patch: ResumeProfilePatch | None,
    parsed_resume: ParsedResume | None,
    parsed_job_description: ParsedJobDescription | None,
    requested_target_position: str | None,
) -> AutoProfileDraft:
    """Merge trusted structured snapshots into a reviewable draft without persisting it."""
    draft = ProfileUpdate.model_validate(dict(existing_profile or {}))
    values = draft.model_dump()
    sources: dict[str, ProfileFieldSource] = {
        field: "manual" for field, value in values.items() if _has_value(value)
    }

    if resume_patch is not None:
        for field, value in resume_patch.model_dump(exclude_none=True).items():
            if _has_value(value):
                values[field] = value
                sources[field] = "resume"

    if parsed_job_description is not None:
        values["target_position"] = parsed_job_description.target_position
        sources["target_position"] = "job_description"

    if requested_target_position:
        values["target_position"] = requested_target_position
        sources["target_position"] = "request"

    merged = ProfileUpdate.model_validate(values)
    completeness, missing_fields = calculate_profile_completeness(merged)
    warnings = _build_warnings(
        completeness=completeness,
        missing_fields=missing_fields,
        merged=merged,
        parsed_resume=parsed_resume,
        parsed_job_description=parsed_job_description,
        has_resume=resume_patch is not None or parsed_resume is not None,
    )
    return AutoProfileDraft(
        profile_patch=merged,
        completeness=completeness,
        auto_summary=_build_auto_summary(merged, parsed_resume),
        warnings=warnings,
        missing_fields=missing_fields,
        field_sources=sources,
    )


def calculate_profile_completeness(profile: ProfileUpdate) -> tuple[int, list[str]]:
    values = profile.model_dump()
    score = sum(
        weight
        for field, weight in PROFILE_COMPLETENESS_WEIGHTS.items()
        if _has_value(values.get(field))
    )
    missing_fields = [
        field for field in PROFILE_COMPLETENESS_WEIGHTS if not _has_value(values.get(field))
    ]
    return score, missing_fields


def _build_warnings(
    *,
    completeness: int,
    missing_fields: list[str],
    merged: ProfileUpdate,
    parsed_resume: ParsedResume | None,
    parsed_job_description: ParsedJobDescription | None,
    has_resume: bool,
) -> list[str]:
    warnings: list[str] = []
    if completeness < LOW_COMPLETENESS_THRESHOLD:
        labels = "、".join(PROFILE_FIELD_LABELS[field] for field in missing_fields)
        warnings.append(
            f"画像完整度低于 {LOW_COMPLETENESS_THRESHOLD}%，建议补充：{labels}。"
        )
    if not has_resume:
        warnings.append("未提供已解析简历，项目深挖将主要依据现有手动画像。")

    if parsed_job_description is not None:
        candidate_skill_text = " ".join(
            [merged.skills or "", *(parsed_resume.skills if parsed_resume else [])]
        )
        missing_required_skills = [
            skill
            for skill in parsed_job_description.must_have_skills
            if not _skill_is_mentioned(skill, candidate_skill_text)
        ]
        if missing_required_skills:
            skills = "、".join(missing_required_skills[:8])
            warnings.append(
                f"简历和现有画像中未明确体现 JD 硬性技能：{skills}；请仅按真实经历补充。"
            )
        warnings.extend(
            f"JD 风险提示：{risk}" for risk in parsed_job_description.risk_points[:8]
        )
    return _deduplicate(warnings)


def _build_auto_summary(profile: ProfileUpdate, parsed_resume: ParsedResume | None) -> str | None:
    parts: list[str] = []
    if profile.target_position:
        parts.append(f"目标岗位：{profile.target_position}")
    if profile.experience_years is not None:
        parts.append(f"经验：{profile.experience_years} 年")
    if profile.skills:
        skill_summary = re.sub(r"\s+", " ", profile.skills).strip()
        parts.append(f"技能：{skill_summary[:240]}")
    project_count = len(parsed_resume.projects) if parsed_resume else 0
    if project_count:
        parts.append(f"简历项目：{project_count} 个")
    elif profile.projects:
        parts.append("已包含项目经历")
    return "；".join(parts) or None


def _skill_is_mentioned(required_skill: str, candidate_text: str) -> bool:
    required = _normalize_skill(required_skill)
    candidate = _normalize_skill(candidate_text)
    return bool(required) and required in candidate


def _normalize_skill(value: str) -> str:
    return "".join(char for char in value.casefold() if char.isalnum() or "\u4e00" <= char <= "\u9fff")


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _deduplicate(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item.strip()))
