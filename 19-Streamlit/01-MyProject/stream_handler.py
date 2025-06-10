import streamlit as st


def get_current_tool_message(tool_args, tool_call_id):
    if tool_call_id:
        for tool_arg in tool_args:
            if tool_arg["tool_call_id"] == tool_call_id:
                return tool_arg
        return None
    else:
        return None


def format_search_result(results):
    import json

    results = json.loads(results)

    answer = ""
    for result in results:
        answer += f'**[{result["title"]}]({result["url"]})**\n\n'
        answer += f'{result["content"]}\n\n'
        answer += f'신뢰도: {result["score"]}\n\n'
        answer += "\n-----\n"
    return answer


def stream_handler(streamlit_container, agent_executor, inputs, config):
    # 결과를 저장할 리스트 초기화
    tool_args = []
    agent_answer = ""
    agent_message = None  # agent_message 변수를 미리 선언

    container = streamlit_container.container()
    with container:
        for chunk_msg, metadata in agent_executor.stream(
            inputs, config, stream_mode="messages"
        ):
            if hasattr(chunk_msg, "tool_calls") and chunk_msg.tool_calls:
                # 도구 호출 결과 초기화
                tool_arg = {
                    "tool_name": "",
                    "tool_result": "",
                    "tool_call_id": chunk_msg.tool_calls[0]["id"],
                }
                # 도구 이름 저장
                tool_arg["tool_name"] = chunk_msg.tool_calls[0]["name"]
                if tool_arg["tool_name"]:
                    tool_args.append(tool_arg)

            if hasattr(chunk_msg, "tool_call_chunks") and chunk_msg.tool_call_chunks:
                if len(chunk_msg.tool_call_chunks) > 0:  # None 체크 추가
                    # 도구 호출 인자 누적
                    chunk_msg.tool_call_chunks[0]["args"]

            if metadata["langgraph_node"] == "tools":
                # 도구 실행 결과 저장
                current_tool_message = get_current_tool_message(
                    tool_args, chunk_msg.tool_call_id
                )
                if current_tool_message:
                    current_tool_message["tool_result"] = chunk_msg.content
                    with st.status(f'✅ {current_tool_message["tool_name"]}'):
                        if current_tool_message["tool_name"] == "web_search":
                            st.markdown(
                                format_search_result(
                                    current_tool_message["tool_result"]
                                )
                            )

            if metadata["langgraph_node"] == "agent":
                if chunk_msg.content:
                    if agent_message is None:
                        agent_message = st.empty()
                    # 에이전트 메시지 누적
                    agent_answer += chunk_msg.content
                    agent_message.markdown(agent_answer)

        return container, tool_args, agent_answer
