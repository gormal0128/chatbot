import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 🔑 API 키 세팅
# 기존의 이 코드를 통째로 지우세요!
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# 👇 대신 이 코드를 복사해서 넣으세요!
# with st.sidebar:
#    st.title("🔑 챗봇 시작하기")
#    st.caption("제미나이 API 키를 입력해야 챗봇이 깨어납니다.")
#    user_api_key = st.text_input("Gemini API Key", type="password")
#
# if not user_api_key:
#    st.warning("👈 왼쪽 사이드바에 API 키를 입력하시면 채팅창이 열립니다!")
#    st.stop() # 키를 넣기 전까지는 챗봇을 잠깐 멈춰둡니다.
#
# 사용자가 화면에 입력한 키를 챗봇 뇌에 꽂아줍니다!
# os.environ["GOOGLE_API_KEY"] = user_api_key

# 🎨 1. 웹페이지 기본 설정
st.set_page_config(page_title="사내 규정 챗봇", page_icon="🤖")
st.title("사내 규정 & 지침 챗봇")
st.caption("궁금한 회사 규정을 물어보세요!")

# 🧠 2. 챗봇 뇌(DB) 불러오기 (한 번만 불러오고 캐시에 저장하여 속도 향상)
@st.cache_resource
def load_chatbot():
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    # 이미 만들어둔 chroma_db 폴더를 즉시 불러옵니다! (기다림 없음)
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    prompt = PromptTemplate.from_template(
        "당신은 회사 규정 및 지침을 안내하는 전문 AI 비서입니다.\n"
        "아래 제공된 [관련 규정]만을 바탕으로 사용자의 [질문]에 친절하게 답변해주세요.\n"
        "규정에 없는 내용은 지어내지 말고 '제공된 지침에서는 해당 내용을 찾을 수 없습니다'라고 답변하세요.\n\n"
        "[관련 규정]\n{context}\n\n"
        "[질문]\n{question}\n\n"
        "답변:"
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return qa_chain

qa_chain = load_chatbot()

# 💬 3. 채팅 기록 저장소 세팅
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 채팅 기록을 화면에 띄워주기
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ⌨️ 4. 사용자 입력 창 및 답변 출력
if prompt := st.chat_input("규정에 대해 질문해 보세요..."):
    # 사용자가 입력한 질문을 화면에 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 챗봇이 생각하고 답변을 출력하는 부분
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("규정집을 열심히 뒤적이는 중... 📖"):
            # DB를 검색해서 답변 생성
            response = qa_chain.invoke(prompt)
            message_placeholder.markdown(response)
    
    # 생성된 답변을 기록에 저장
    st.session_state.messages.append({"role": "assistant", "content": response})
