import asyncio
import sys
from pathlib import Path

from openai import APIConnectionError


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from langchain_core.messages import AIMessage, HumanMessage

from app.agents.graph import interview_graph
from app.agents.state import create_initial_state


async def main() -> None:
    # 1. 创建一份模拟面试状态，相当于一次面试会话的上下文。
    state = create_initial_state(
        user_id=1,
        session_id=1,
        target_position="Python 后端工程师",
        difficulty="medium",
        profile={
            "education": "本科",
            "major": "计算机科学与技术",
            "experience_years": 1,
            "skills": "Python, FastAPI, MySQL, Vue",
            "projects": "AI 模拟面试系统，负责用户登录、画像管理、面试会话接口",
        },
    )

    # 2. 模拟用户已经说过一句话，LangGraph 会把这条消息作为历史上下文。
    state["messages"].append(
        HumanMessage(content="我做过一个 AI 模拟面试系统，后端主要用了 FastAPI 和 MySQL。")
    )
    # 3. 统一面试图的 start 动作：生成第一道问题。
    result = await interview_graph.ainvoke(state)

    # 4. 模拟用户回答当前问题。真实接口里，这一步来自前端提交回答。
    result["messages"].append(
        HumanMessage(content="我使用 bcrypt 对密码进行哈希存储，登录成功后返回 JWT，后续接口通过 Bearer Token 鉴权。")
    )
    result["action"] = "answer"

    # 5. 统一面试图的 answer 动作：一次完成评分、追问判断和下一题生成。
    result = await interview_graph.ainvoke(result)

    # 7. 打印完整消息历史，方便你看 add_messages 是否自动合并。
    print("\n===== messages =====")
    for message in result["messages"]:
        if isinstance(message, HumanMessage):
            print(f"User: {message.content}")
        elif isinstance(message, AIMessage):
            print(f"AI: {message.content}")
        else:
            print(f"{message.type}: {message.content}")

    # 8. 打印当前状态里的关键字段。
    print("\n===== state =====")
    print(f"current_question: {result['current_question']}")
    print(f"current_dimension: {result['current_dimension']}")

    print("\n===== score =====")
    print(result["scores"][-1])

    print("\n===== followup =====")
    print(result["followup_decision"])


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except APIConnectionError as exc:
        print("模型 API 连接失败，请检查网络、代理、OPENAI_API_BASE 和 OPENAI_API_KEY。")
        raise exc
