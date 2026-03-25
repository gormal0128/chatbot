import streamlit as st
import os
import re  # 🛠️ [수정 포인트 1] 정규표현식 라이브러리 추가
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# 🛠️ [수정 포인트 2] HWP 표 텍스트 공백 청소기 함수 추가
def clean_hwp_text(text):
    # 연속된 공백(스페이스, 탭, 줄바꿈 등)을 단 하나의 스페이스로 압축합니다.
    # 이렇게 하면 HWP 표에서 발생한 넓은 자간이 어느 정도 정상화됩니다.
    return re.sub(r'\s+', ' ', text).strip()

# 🔑 API 키 세팅 (비밀 금고에서 가져오기)
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# 🎨 1. 웹페이지 기본 설정
st.set_page_config(page_title="사내 규정 챗봇", page_icon="🤖", layout="wide")
st.title("🤖 사내 규정 & 지침 챗봇")
st.caption("궁금한 회사 규정을 물어보세요! 이전 대화도 기억합니다. 🧠")

# 🧠 2. 챗봇 뇌(DB) 및 LLM 불러오기
@st.cache_resource
def load_chatbot_components():
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
    return retriever, llm

retriever, llm = load_chatbot_components()

# 📝 프롬프트 템플릿
prompt_template = PromptTemplate.from_template(
    "당신은 회사 규정 및 지침을 안내하는 전문 AI 비서입니다.\n"
    "아래 제공된 [이전 대화 기록]의 문맥을 파악하고, [관련 규정]만을 바탕으로 사용자의 [질문]에 친절하게 답변해주세요.\n"
    "규정에 없는 내용은 지어내지 말고 '제공된 지침에서는 해당 내용을 찾을 수 없습니다'라고 답변하세요.\n\n"
    "[이전 대화 기록]\n{history}\n\n"
    "[관련 규정]\n{context}\n\n"
    "[질문]\n{question}\n\n"
    "답변:"
)

# 💾 3. 세션(Session) 저장소 세팅
if "messages" not in st.session_state:
    st.session_state.messages = []
if "stats" not in st.session_state:
    st.session_state.stats = {} 

# 📊 [고도화 1] 관리자용 통계 대시보드 (사이드바)
with st.sidebar:
    st.header("📈 관리자 통계")
    st.caption("오늘 직원들이 많이 찾은 규정")
    if st.session_state.stats:
        sorted_stats = sorted(st.session_state.stats.items(), key=lambda x: x[1], reverse=True)
        for q, count in sorted_stats:
            st.write(f"- {q} ({count}회)")
    else:
        st.write("아직 질문 기록이 없습니다.")
    
    st.divider()
    if st.button("🗑️ 대화 기록 초기화"):
        st.session_state.messages = []
        st.rerun()

# 🔘 [고도화 2] 자주 묻는 질문(FAQ) 퀵 버튼
st.write("💡 **자주 묻는 질문(FAQ)**")
col1, col2, col3 = st.columns(3)
faq_clicked = None
if col1.button("🌴 연차/휴가 규정 알려줘"): faq_clicked = "연차 및 휴가 관련 규정을 요약해줘."
if col2.button("⏰ 지각 시 처리 방법은?"): faq_clicked = "지각 시 불이익이나 처리 방법이 어떻게 돼?"
if col3.button("💳 출장비/식대 한도"): faq_clicked = "출장 시 식대와 숙박비 한도를 알려줘."

# 💬 기존 채팅 기록을 화면에 띄워주기
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ⌨️ 4. 사용자 질문 입력 및 답변 생성
user_input = st.chat_input("규정에 대해 질문해 보세요...")
prompt = faq_clicked if faq_clicked else user_input

if prompt:
    st.session_state.stats[prompt] = st.session_state.stats.get(prompt, 0) + 1
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("규정집을 꼼꼼히 확인하는 중... 📖"):
            try:
                # 1. DB에서 관련 문서 찾아오기
                docs = retriever.invoke(prompt)
                
                # 🛠️ [수정 포인트 3] 검색된 문서의 더러운 공백을 싹 청소합니다!
                clean_docs_content = [clean_hwp_text(doc.page_content) for doc in docs]
                context_text = "\n\n".join(clean_docs_content)
                
                # 2. [고도화 3] 이전 대화 기록 묶기
                history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-5:-1]])
                
                # 3. LLM에게 질문 던지기 (이제 LLM도 깔끔한 텍스트를 읽습니다)
                final_prompt = prompt_template.format(history=history_text, context=context_text, question=prompt)
                ai_message = llm.invoke(final_prompt)
                
                # 4. 답변과 출처 분리해서 출력하기
                
                # 먼저 LLM의 답변만 화면에 뿌려줍니다 (만약 스트리밍을 원하시면 st.write_stream 사용)
                message_placeholder.markdown(ai_message.content) 
                
                # 출처 텍스트를 따로 조립합니다
                source_text = "\n\n---\n**🔍 참고한 규정 원문**\n"
                for text in clean_docs_content:
                    snippet = text[:150] 
                    source_text += f"> *...{snippet}...*\n\n"

                # 조립된 출처는 스트리밍이 아닌 일반 마크다운으로 '한 번에' 화면에 덧붙입니다.
                st.markdown(source_text)
                
                # 세션에 저장할 최종 기록은 둘을 합쳐서 저장합니다.
                response = ai_message.content + source_text

            except Exception as e:
                # 🚨 [고도화 5] 에러 메시지 예쁘게 포장하기
                error_msg = str(e).lower()
                if "429" in error_msg or "resource_exhausted" in error_msg or "quota" in error_msg:
                    response = "🚨 **하루 사용량이 모두 소요되었습니다.** (무료 할당량 초과)\n\n죄송합니다. 오늘은 챗봇이 너무 많은 질문을 받아 지쳤습니다. 내일 다시 이용해 주시거나, 관리자에게 문의해 주세요!"
                else:
                    response = f"🚨 시스템 오류가 발생했습니다. 잠시 후 다시 시도해 주세요. (에러: {e})"
                
                message_placeholder.error(response)
    
    # 생성된 답변을 기록에 저장
    st.session_state.messages.append({"role": "assistant", "content": response})
