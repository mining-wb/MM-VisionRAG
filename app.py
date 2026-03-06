import streamlit as st
import requests

# ====== 界面布局层 ======
st.sidebar.title("MM-VisionRAG")
st.sidebar.caption("多模态检索增强，看图+文档问答。")
st.sidebar.divider()
st.sidebar.subheader("上传参考文档")
upload_doc = st.sidebar.file_uploader("PDF / TXT", type=["pdf", "txt"], key="upload_doc", label_visibility="collapsed")
st.sidebar.subheader("上传图片")
st.sidebar.file_uploader("图片", type=["png", "jpg", "jpeg"], key="upload_img", label_visibility="collapsed")

# 文档上传后发到后端做解析与向量化，同一文件只传一次
if upload_doc is not None:
    last = st.session_state.get("last_uploaded_name")
    if last != upload_doc.name:
        try:
            r = requests.post(
                "http://127.0.0.1:8000/api/v1/upload",
                files={"file": (upload_doc.name, upload_doc.getvalue(), upload_doc.type or "application/octet-stream")},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            st.session_state["last_uploaded_name"] = upload_doc.name
            st.sidebar.success(f"已入库：{data.get('chunk_count', 0)} 块")
        except Exception as e:
            st.sidebar.error(f"上传失败：{e}")

# ====== 状态管理层 ======
# 用 session_state 存对话，否则每次交互重跑脚本，之前的聊天就没了
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ====== 前后端通信层 ======

if prompt := st.chat_input("输入问题，回车发送"):
    # 用户问题先追加到记录并渲染
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # POST 到本地 FastAPI，拿回复
    try:
        r = requests.post(
            "http://127.0.0.1:8000/api/v1/chat",
            json={"question": prompt},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        answer = data.get("answer", "（无回复）")
    except Exception as e:
        answer = f"请求失败：{e}"

    # 解析 JSON，把助手回复追加到记录并展示
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
