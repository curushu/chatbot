import pandas as pd
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from sqlalchemy import create_engine, text, MetaData, Table
import re
from typing import Dict, Any
import json
import os
from datetime import datetime
import requests
from dotenv import load_dotenv
import streamlit as st
from passlib.context import CryptContext

# 환경 변수 로드
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DB_ROOT_PASSWORD = os.environ.get("root")

# DB 연결 설정
DB_URL = f"mysql+pymysql://root:{DB_ROOT_PASSWORD}@localhost:3306/test"
engine = create_engine(DB_URL)

# 비밀번호 해싱
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 테스트용 사용자 DB
users_db = {
    "testuser": {
        "username": "testuser",
        "hashed_password": pwd_context.hash("testpassword"),
    }
}

# 사용자 인증 함수
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# CustomSQLChatMessageHistory 클래스
class CustomSQLChatMessageHistory(SQLChatMessageHistory):
    def add_message(self, message: BaseMessage) -> None:
        if isinstance(message, HumanMessage):
            role = "human"
        elif isinstance(message, AIMessage):
            role = "ai"
        else:
            role = "unknown"

        with self._make_sync_session() as session:
            metadata = MetaData()
            message_table = Table("message_store", metadata, autoload_with=session.bind)

            insert_stmt = message_table.insert().values(
                session_id=self.session_id,
                role=role,
                message=message.content,
            )

            session.execute(insert_stmt)
            session.commit()

    def add_ai_message(self, message: str) -> None:
        self.add_message(AIMessage(content=message))

    def add_user_message(self, message: str) -> None:
        self.add_message(HumanMessage(content=message))

    def get_chat_history(self):
        try:
            with engine.connect() as conn:
                query = f"""
                SELECT role, message, created_at
                FROM message_store 
                WHERE session_id = '{self.session_id}'
                ORDER BY created_at DESC
                LIMIT 100
                """
                print(query)
                result = pd.read_sql(text(query), conn)
                
                if not result.empty:
                    result = result.iloc[::-1]
                    formatted_history = []
                    for _, row in result.iterrows():
                        timestamp = pd.to_datetime(row['created_at']).strftime('%Y년 %m월 %d일 %H:%M')
                        if row['role'] == 'human':
                            formatted_history.append(f"채팅시간: [{timestamp}] 세션:[{self.session_id}] 사용자: {row['message']}")
                        elif row['role'] == 'ai':
                            formatted_history.append(f"채팅시간: [{timestamp}] 세션:[{self.session_id}] 챗봇: {row['message']}")
                    return "\n".join(formatted_history)
                else:
                    return "대화 기록이 없습니다."
        except Exception as e:
            print(f"❌ 채팅 히스토리 가져오기 오류: {str(e)}")
            return ""

# EnhancedQueryGenerator 클래스
class EnhancedQueryGenerator:
    def __init__(self):
        if not GEMINI_API_KEY or GEMINI_API_KEY == "your-gemini-api-key-here":
            raise ValueError("❌ Gemini API 키가 설정되지 않았습니다.")

        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        self.headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY
        }

    def generate_query(self, question: str, schema_info: str, chat_history: str = "") -> str:
        cur_time = datetime.now().strftime('%Y.%m.%d- %H:%M:%S')
        prompt = f"""
        당신은 MySQL 전문가입니다. 이전 대화 흐름을 보고 다음 데이터베이스 스키마를 기반으로 
        사용자의 질문에 적절한 SQL 쿼리를 생성하세요.
        
        **이전 대화 내용**:
        {chat_history}

        데이터베이스 스키마 정보:
        {schema_info}

        질문: {question}
        
        현시각: {cur_time}

        **규칙**:
        - 이전 대화 내용을 반드시 먼저 확인하고 sql문에 반영해주세요 (특히 날짜)
        - SQL 쿼리만 반환하세요. 설명은 필요하지 않습니다.
        - `SELECT` 문으로 시작하고 세미콜론(;)으로 끝나야 합니다.
        - SQL 인젝션 공격을 방지하기 위해 파라미터 바인딩을 사용하지 마세요.
        - 정확한 값 매칭은 `=` 연산자를 사용하세요.
        - 유사 검색은 `LIKE '%키워드%'` 형식을 사용하세요.
        - LIKE문에 모든 단어를 넣지 마세요.
        - LIKE문에 최대한 and 대신 or을 사용해 넓은 범위를 수색해주세요.
        - 날짜를 조건문에 사용할땐 date(`날짜 문자열`)이 아닌 `날짜 문자열` 그대로 사용하세요
        - 예시 쿼리:
        SELECT * FROM tsla_stock WHERE currentdate = '2010-06-29';
        - 날짜 비교 시에는 BETWEEN 연산자를 사용하고, 날짜 문자열은 작은따옴표('')로 묶으세요.
        - 날짜를 비교할 때 DATE_SUB 함수를 사용하지 마세요.
        - 서브쿼리는 되도록이면 in을 사용하세요

        결과:
        """

        response = self._call_gemini_api(prompt)
        return self._extract_sql_query(response)

    def generate_answer(self, question: str, query: str, result: any, chat_history: str="") -> str:
        cur_time = datetime.now().strftime('%Y.%m.%d- %H:%M:%S')
        if isinstance(result, pd.DataFrame):
            result_str = result.to_json(orient="records")
        elif isinstance(result, str):
            return result
        else:
            result_str = json.dumps(result, ensure_ascii=False)

        prompt = f"""
        사용자의 질문에 답변하세요. 이전 대화 내용을 고려하세요.

        **이전 대화 내용**:
        {chat_history}

        **질문**: {question}
        **실행된 쿼리**: {query}
        **쿼리 결과**:
        {result_str}
        **현시각**:{cur_time}

        **규칙**:
        - 이전 대화 내용에서 사용자가 언급한 정보를 우선적으로 활용하세요.
        - 이전 대화 내용에서 정보를 찾을 수 없는 경우에만 쿼리 결과를 활용하세요.
        - 결과를 자연스러운 한국어로 설명하세요.
        - 숫자 데이터가 있다면 적절한 단위를 포함하세요.
        - 이전 대화 내용이 없으면 쿼리 결과를 기반으로 답변하세요.
        - 쿼리 결과을 기반으로도 결과가 없다면 주어진 정보가 아닌 기존 Gemini API로써의 답변으로 답변하세요.
        - 사용자가 쿼리, 이전 대화 기록이 아닌 기본 답변을 원한다고 요청시 그것 또한 Gemini API로써의 답변으로 답변하세요.
        - date는 반드시 `xxxx년 xx월 xx일`로 출력하세요
        """

        return self._call_gemini_api(prompt)

    def _call_gemini_api(self, prompt: str) -> str:
        try:
            data = {"contents": [{"parts": [{"text": prompt}]}]}
            response = requests.post(self.api_url, headers=self.headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            print(f"❌ Gemini API 호출 오류: {str(e)}")
            return f"❌ Gemini API 호출 오류: {str(e)}"

    @staticmethod
    def _extract_sql_query(response: str) -> str:
        response = response.replace("```sql", "").replace("```", "").strip()
        match = re.search(r"SELECT\s+.*?\s+;", response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(0).strip()
        else:
            print(f"❌ SQL 쿼리 추출 실패: {response}")
            return response.strip()

# DB 스키마 정보 가져오기
def get_schema_info():
    with engine.connect() as conn:
        tables = pd.read_sql("SHOW TABLES", conn)
        schema_info = []

        for table in tables.iloc[:, 0]:
            columns = pd.read_sql(f"DESCRIBE {table}", conn)
            schema_info.append(f"테이블: {table}")
            schema_info.append("컬럼:")
            for _, row in columns.iterrows():
                schema_info.append(f"- {row['Field']} ({row['Type']})")
            schema_info.append("")

        return "\n".join(schema_info)

# SQL 쿼리 실행
def execute_query(query):
    try:
        with engine.connect() as conn:
            result = pd.read_sql(text(query), conn)
            return result
    except Exception as e:
        print(f"❌ 쿼리 실행 중 오류 발생: {str(e)}")
        return f"❌ 쿼리 실행 중 오류 발생: {str(e)}"

# 질문 처리
def process_question(question: str, session_id: str):
    schema_info = get_schema_info()
    query_generator = EnhancedQueryGenerator()
    chat_message_history = CustomSQLChatMessageHistory(session_id=session_id, connection_string=DB_URL)
    chat_history = chat_message_history.get_chat_history()

    query = query_generator.generate_query(question, schema_info, chat_history)
    result = execute_query(query)
    answer = query_generator.generate_answer(question, query, result, chat_history)

    chat_message_history.add_user_message(question)
    chat_message_history.add_ai_message(answer)
    return answer, chat_history

# Streamlit 앱
def main():
    # Streamlit 페이지 설정
    st.set_page_config(page_title="RAG DB 챗봇", page_icon="🤖", layout="centered")

    # CSS 스타일링 (Gradio 스타일 반영)
    st.markdown("""
    <style>
    .stTextInput > div > input {
        border-radius: 8px;
        padding: 12px;
        font-size: 1rem;
        border: 1px solid #d1d5db;
    }
    .stButton > button {
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        background: linear-gradient(90deg, #3b82f6, #6366f1);
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .chat-message-user {
        background: linear-gradient(90deg, #3b82f6, #6366f1);
        color: white;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chat-message-bot {
        background: #f1f5f9;
        color: #1f2937;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .header {
        text-align: center;
        margin-bottom: 20px;
    }
    .header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a8a;
    }
    .header p {
        font-size: 1.1rem;
        color: #4b5563;
    }
    </style>
    """, unsafe_allow_html=True)

    # 세션 상태 초기화
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.chat_history = []

    # 로그인 페이지
    if not st.session_state.logged_in:
        st.markdown("""
        <div class="header">
            <h1>RAG DB 챗봇</h1>
            <p>SQL 기반 데이터베이스 질의를 처리하는 AI 챗봇입니다.</p>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("로그인")
        username = st.text_input("사용자 이름", placeholder="testuser")
        password = st.text_input("비밀번호", type="password", placeholder="testpassword")
        login_button = st.button("로그인")

        if login_button:
            user = users_db.get(username)
            if user and verify_password(password, user["hashed_password"]):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("로그인 성공!")
                st.rerun()
            else:
                st.error("잘못된 사용자 이름 또는 비밀번호입니다.")
    else:
        # 챗봇 페이지
        st.markdown("""
        <div class="header">
            <h1>RAG DB 챗봇</h1>
            <p>질문을 입력하여 데이터베이스와 대화하세요!</p>
        </div>
        """, unsafe_allow_html=True)

        # 로그아웃 버튼
        if st.button("로그아웃"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.chat_history = []
            st.rerun()

        # 채팅 히스토리 표시
        st.subheader("대화 기록")
        chat_container = st.container()
        with chat_container:
            for message, response in st.session_state.chat_history:
                st.markdown(f'<div class="chat-message-user">사용자: {message}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-message-bot">챗봇: {response}</div>', unsafe_allow_html=True)

        # 질문 입력
        question = st.text_input("질문을 입력하세요", placeholder="예: 마지막 대화 날짜는 언제인가요?", key="question")
        if st.button("전송"):
            if question.strip():
                answer, chat_history = process_question(question, session_id=st.session_state.username)
                st.session_state.chat_history.append((question, answer))
                st.rerun()
            else:
                st.error("질문을 입력하세요.")

if __name__ == "__main__":
    main()