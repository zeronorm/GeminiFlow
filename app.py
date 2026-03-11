# 请使用 conda 激活你的开发环境
# conda activate dev
# 运行命令: streamlit run app.py

import streamlit as st
import requests

SERVER_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(page_title="Gemini Chat", page_icon="🤖")

st.title("GeminiFlow Chat")

# 初始化 session state 用于存储对话历史和上下文ID
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_ids" not in st.session_state:
    st.session_state.session_ids = {
        "conversation_id": None,
        "response_id": None,
        "choice_id": None
    }

def clear_session():
    """清空当前会话状态，重置为新会话"""
    st.session_state.messages = []
    st.session_state.session_ids = {
        "conversation_id": None,
        "response_id": None,
        "choice_id": None
    }

# 侧边栏
with st.sidebar:
    st.header("控制台")
    st.button("➕ 创建新会话 (New Session)", on_click=clear_session, use_container_width=True)
    st.divider()
    model = st.selectbox("选择模型", ["gemini-3-pro", "gemini-3-pro-thinking", "gemini-2.5-pro", "gemini-2.5-flash"], index=0)

    st.divider()
    with st.expander("🛠️ 调试信息 (Debug)", expanded=False):
        st.write("当前内存中的 Context IDs:")
        st.json(st.session_state.session_ids)
        if st.session_state.messages:
            st.write("对话轮数:", len(st.session_state.messages) // 2)

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # 如果有图片则渲染
        if message.get("images"):
            for img in message["images"]:
                if img.startswith("data:image"):
                    st.image(img)
                else:
                    st.image(f"data:image/jpeg;base64,{img}")

# 接收用户输入
if prompt := st.chat_input("说点什么吧..."):
    # 用户消息入栈
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 显示模型回复
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("正在思考...")
        
        # 组装发给 server.py 的请求数据
        payload = {
            "prompt": prompt,
            "model": model
        }
        
        # 如果有上下文 IDs，带上它们以维持多轮对话记忆
        if st.session_state.session_ids["conversation_id"]:
            payload["conversation_id"] = st.session_state.session_ids["conversation_id"]
            payload["response_id"] = st.session_state.session_ids["response_id"]
            payload["choice_id"] = st.session_state.session_ids["choice_id"]

        try:
            # 发起 POST 请求到本地 server.py
            response = requests.post(
                SERVER_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                proxies={"http": None, "https": None}  # 防止系统代理拦截本地请求导致 405 或 502
            )
            
            if response.status_code == 200:
                data = response.json()
                bot_reply = data.get("text", "")
                images = data.get("images", [])
                
                # 提取并更新最新的上下文 IDs
                st.session_state.session_ids["conversation_id"] = data.get("conversation_id")
                st.session_state.session_ids["response_id"] = data.get("response_id")
                st.session_state.session_ids["choice_id"] = data.get("choice_id")
                
                # 更新占位符中的文本
                message_placeholder.markdown(bot_reply)
                
                # 渲染服务器传回的 base64 图片
                for img in images:
                    if img.startswith("data:image"):
                        st.image(img)
                    else:
                        st.image(f"data:image/jpeg;base64,{img}")
                
                # 将回复保存到历史记录
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": bot_reply, 
                    "images": images
                })
            else:
                error_msg = f"服务器错误 {response.status_code}: {response.text}"
                message_placeholder.markdown(f"**错误**: {error_msg}")
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
        except Exception as e:
            error_msg = f"无法连接到服务器: {str(e)} \n请确保 `python server.py` 正在后台运行。"
            message_placeholder.markdown(f"**请求失败**: {error_msg}")
            st.session_state.messages.append({"role": "assistant", "content": error_msg})