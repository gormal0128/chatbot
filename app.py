import streamlit as st
import os
import re  # 🛠️ [수정 포인트 1] 정규표현식 라이브러리 추가
import datetime # 📊 시간 기록용
import gspread  # 📊 구글 시트 조작용
from google.oauth2.service_account import Credentials # 🔐 구글 인증용
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# 🛠️ [수정 포인트 2] HWP 표 텍스트 공백 청소기 함수 추가
def clean_hwp_text(text):
    # 연속된 공백(스페이스, 탭, 줄바꿈 등)을 단 하나의 스페이스로 압축합니다.
    # 이렇게 하면 HWP 표에서 발생한 넓은 자간이 어느 정도 정상화됩니다.
    return re.sub(r'\s+', ' ', text).strip()

# ==========================================
# 📊 [고도화 6] 구글 시트 실시간 로깅 함수
# ==========================================
def log_to_sheet(question, answer):
    try:
        # 1. Streamlit Secrets에서 로봇 직원 신분증(JSON) 가져오기
        credentials_dict = dict(st.secrets["gcp_service_account"])
        
        # 2. 구글 시트 및 드라이브 접근 권한 설정
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        
        # 3. 로봇 직원 로그인
        creds = Credentials.from_service_account_info(credentials_dict, scopes=scopes)
        client = gspread.authorize(creds)
        
        # 4. 엑셀 파일 열기 (⚠️ 주의: 구글 시트 파일 이름이 완벽하게 일치해야 합니다!)
        sheet = client.open("사내챗봇 QnA 로그").sheet1
        
        # 5. 현재 시간 기록 (한국 시간 기준)
        now = (datetime.datetime.utcnow() + datetime.timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S")
        
        # 6. 맨 아래 빈 줄에 [시간, 질문, 답변] 쏙 집어넣기
        sheet.append_row([now, question, answer])
        
    except Exception as e:
        # 로그 저장에 실패해도 챗봇은 정상 작동하도록 에러를 화면에 띄우지 않습니다.
        print(f"로그 저장 실패: {e}") 

# 🔑 API 키 세팅 (비밀 금고에서 가져오기)
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# 🎨 1. 웹페이지 기본 설정
st.set_page_config(page_title="사내 규정 챗봇", page_icon="🤖", layout="wide")
st.title("RAPA 지침 챗봇 (v0.1 테스트)")
st.caption("궁금한 회사 지침을 물어보세요!")

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

# 🔘 [고도화 2] 자주 묻는 질문(FAQ) 퀵 버튼
st.write("💡 **자주 묻는 질문(FAQ)**")
col1, col2, col3 = st.columns(3)
faq_clicked = None
if col1.button("병가 규정 알려줘"): faq_clicked = "병가 관련 규정을 요약해줘."
if col2.button("200만원 이상의 연구반회의 전결"): faq_clicked = "연구반 회의 지출이 200만원 이상일때 전결권자는 어떻게 돼?"
if col3.button("출장비/식대 한도"): faq_clicked = "출장 시 식대와 숙박비 한도를 알려줘."

# 💬 기존 채팅 기록을 화면에 띄워주기
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ⌨️ 4. 사용자 질문 입력 및 답변 생성
user_input = st.chat_input("지침 중 궁금한 내용을 질문해 보세요...")
prompt = faq_clicked if faq_clicked else user_input

if prompt:
    st.session_state.stats[prompt] = st.session_state.stats.get(prompt, 0) + 1
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("지침을 확인하고 있습니다. 잠시만 기다려 주세요..."):
            try:
                # 1. DB에서 관련 문서 찾아오기
                docs = retriever.invoke(prompt)
                
                # 🧹 [공백 청소기 작동] 검색된 문서의 더러운 공백 압축
                clean_docs_content = [clean_hwp_text(doc.page_content) for doc in docs]
                context_text = "\n\n".join(clean_docs_content)
                
                # 2. [고도화 3] 이전 대화 기록 묶기
                history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-5:-1]])
                
                # 3. LLM에게 질문 던지기
                final_prompt = prompt_template.format(history=history_text, context=context_text, question=prompt)
                ai_message = llm.invoke(final_prompt)
                
                # ==========================================
                # 🛡️ [철벽 방어] 모든 에러를 막아내는 최종 출력 로직
                # ==========================================
                
                # (1) 리스트 에러 방어: Gemini 답변에서 순수 '텍스트'만 안전하게 추출
                llm_text = ""
                if isinstance(ai_message.content, list):
                    for item in ai_message.content:
                        if isinstance(item, dict) and 'text' in item:
                            llm_text += item['text']
                        elif isinstance
