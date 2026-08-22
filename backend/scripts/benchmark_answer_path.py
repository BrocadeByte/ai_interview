"""回答路径延迟对比基准。

模拟真实 LLM 生成与 RAG 检索耗时，对比老的「评分 → 判断追问 → 生成下一题」三次串行
调用与新的合并节点一次调用，在「不追问」和「追问」两种场景下的：

- 端到端总耗时
- LLM 调用次数
- RAG 检索次数
- 首个可见字符延迟（TTFD，候选人看到第一个问题字符的等待时间）

运行：python scripts/benchmark_answer_path.py
"""
import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import app.agents.nodes.answer_evaluator as answer_evaluator
import app.agents.nodes.answer_pipeline as answer_pipeline
import app.agents.nodes.followup_decider as followup_decider
import app.agents.nodes.question_generator as question_generator
from app.agents.state import create_initial_state
from app.services.llm_stream import (
    reset_stream_delta_callback,
    reset_stream_text_done_callback,
    set_stream_delta_callback,
    set_stream_text_done_callback,
)
from langchain_core.messages import HumanMessage


# ---- 可调的耗时模型（秒），按 DeepSeek + DashScope embedding + Qdrant 的典型量级 ----
LLM_TOTAL = 3.0          # 一次 LLM 调用总生成耗时
LLM_FIRST_TOKEN = 0.4    # 流式首字延迟
RAG_LATENCY = 0.5        # 一次知识检索（embedding HTTP + Qdrant）耗时


class TimedLLM:
    """按角色返回不同 payload，并模拟真实生成耗时。"""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    def _next(self) -> str:
        index = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[index]

    async def ainvoke(self, messages):
        await asyncio.sleep(LLM_TOTAL)  # 非流式：整段生成完才返回
        return SimpleNamespace(content=self._next())

    async def astream(self, messages):
        await asyncio.sleep(LLM_FIRST_TOKEN)
        content = self._next()
        # 把内容切成多段，模拟增量推送，整体跨 LLM_TOTAL。
        n = max(2, len(content) // 8)
        step = max(0.01, (LLM_TOTAL - LLM_FIRST_TOKEN) / n)
        for i in range(n):
            await asyncio.sleep(step)
            start = i * len(content) // n
            end = (i + 1) * len(content) // n
            yield SimpleNamespace(content=content[start:end])


class TimedKnowledge:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, query: str, limit: int = 5) -> str:
        self.calls += 1
        await asyncio.sleep(RAG_LATENCY)
        return "知识库上下文"


def base_state() -> dict:
    state = create_initial_state(
        user_id=1,
        session_id=1,
        target_position="Python 后端工程师",
        difficulty="medium",
        profile={"skills": "Python, FastAPI, MySQL"},
    )
    state["action"] = "answer"
    state["interview_plan"] = [
        {"dimension": "项目经验", "question_count": 2, "weight": 0.25, "focus": "项目复盘"},
        {"dimension": "专业基础", "question_count": 2, "weight": 0.25, "focus": "基础"},
        {"dimension": "系统设计", "question_count": 2, "weight": 0.25, "focus": "架构"},
        {"dimension": "问题排查与协作", "question_count": 2, "weight": 0.25, "focus": "排查"},
    ]
    state["current_question_index"] = 1
    state["current_dimension"] = "项目经验"
    state["current_question"] = "请介绍一个你做过的后端项目。"
    state["messages"] = [HumanMessage(content="我做过一个 FastAPI + MySQL 的面试系统。")]
    return state


def install_delta_recorder() -> tuple[list[float], list[str]]:
    first_delta_at: list[float] = []
    deltas: list[str] = []
    start = time.monotonic()

    async def on_delta(delta: str) -> None:
        if not first_delta_at:
            first_delta_at.append(time.monotonic() - start)
        deltas.append(delta)

    async def on_done() -> None:
        pass

    d_token = set_stream_delta_callback(on_delta)
    t_token = set_stream_text_done_callback(on_done)
    return first_delta_at, deltas, d_token, t_token


async def run_old_chain(followup: bool) -> dict:
    knowledge = TimedKnowledge()
    answer_evaluator.llm = TimedLLM([
        '{"score":82,"sub_scores":{},"reason":"合理。","weaknesses":["缺指标"],"suggestions":["补数据"]}'
    ])
    answer_evaluator.format_knowledge_context = knowledge
    if followup:
        followup_decider.llm = TimedLLM([
            '{"needs_followup":true,"reason":"缺细节。","followup_question":"token 过期如何处理？"}'
        ])
    else:
        followup_decider.llm = TimedLLM([
            '{"needs_followup":false,"reason":"完整。","followup_question":""}'
        ])
    question_generator.llm = TimedLLM([
        '{"question":"你如何设计接口鉴权？","dimension":"项目经验","reason":"继续考察"}'
    ])
    question_generator.format_knowledge_context = knowledge

    first_delta_at, _deltas, d_token, t_token = install_delta_recorder()
    start = time.monotonic()
    try:
        state = base_state()
        state.update(await answer_evaluator.evaluate_answer_node(state))
        state.update(await followup_decider.decide_followup_node(state))
        if not state["followup_decision"]["needs_followup"]:
            state.update(await question_generator.generate_question_node(state))
    finally:
        reset_stream_text_done_callback(t_token)
        reset_stream_delta_callback(d_token)

    return {
        "total": time.monotonic() - start,
        "ttfd": first_delta_at[0] if first_delta_at else None,
        "llm_calls": answer_evaluator.llm.calls + followup_decider.llm.calls + question_generator.llm.calls,
        "rag_calls": knowledge.calls,
    }


async def run_new_pipeline(followup: bool) -> dict:
    knowledge = TimedKnowledge()
    if followup:
        answer_pipeline.llm = TimedLLM([
            '{"needs_followup":true,"decision_reason":"缺细节。","question":"token 过期如何处理？","score":70,"sub_scores":{},"reason":"简略。","weaknesses":["缺细节"],"suggestions":["补充"]}'
        ])
    else:
        answer_pipeline.llm = TimedLLM([
            '{"needs_followup":false,"decision_reason":"完整。","question":"你如何设计接口鉴权？","score":82,"sub_scores":{},"reason":"合理。","weaknesses":["缺指标"],"suggestions":["补数据"]}'
        ])
    answer_pipeline.format_knowledge_context = knowledge

    first_delta_at, _deltas, d_token, t_token = install_delta_recorder()
    start = time.monotonic()
    try:
        await answer_pipeline.answer_pipeline_node(base_state())
    finally:
        reset_stream_text_done_callback(t_token)
        reset_stream_delta_callback(d_token)

    return {
        "total": time.monotonic() - start,
        "ttfd": first_delta_at[0] if first_delta_at else None,
        "llm_calls": answer_pipeline.llm.calls,
        "rag_calls": knowledge.calls,
    }


def fmt(v) -> str:
    return f"{v:.2f}s" if isinstance(v, float) else "—"


def print_row(label: str, r: dict) -> None:
    print(f"  {label:<16} 总耗时={fmt(r['total']):>8}  首字延迟={fmt(r['ttfd']):>8}  LLM={r['llm_calls']}  RAG={r['rag_calls']}")


async def main() -> None:
    print("耗时模型：LLM 总生成 %.1fs / 首字 %.1fs / RAG 检索 %.1fs\n" % (LLM_TOTAL, LLM_FIRST_TOKEN, RAG_LATENCY))

    for scenario, followup in [("不追问（进入下一题）", False), ("追问", True)]:
        print(f"== 场景：{scenario} ==")
        old = await run_old_chain(followup)
        new = await run_new_pipeline(followup)
        print_row("老链路(3节点)", old)
        print_row("新合并节点", new)
        total_save = (1 - new["total"] / old["total"]) * 100 if old["total"] else 0
        ttfd_save = (1 - new["ttfd"] / old["ttfd"]) * 100 if old.get("ttfd") and new.get("ttfd") else 0
        print(f"  -> 总耗时降低 {total_save:.0f}%，首字延迟降低 {ttfd_save:.0f}%\n")


if __name__ == "__main__":
    asyncio.run(main())
