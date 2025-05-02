# Chatbot
## Gemini-api 기반 RAG(Retrieval-Augmented Generation) 모델

## 산출물(PDF)
- [RAG CHATBOT 포트폴리오](https://drive.google.com/file/d/1R6wJkRiTWQFqkvmr1xEABoRiIOWd15rW/view?usp=drive_link)

## 프로젝트 개요

이 프로젝트는 Gemini API와 SQLite DB를 활용하여 RAG(Retrieval-Augmented Generation) 기반 챗봇을 구현한 것입니다.  
사용자의 질문을 바탕으로 LLM이 SQL 문을 자동 생성하고, 해당 결과를 기반으로 답변을 제공합니다.  
또한 최근 100개의 채팅 이력을 반영하여, 자연스럽고 연속성 있는 대화를 지원합니다.

## 기술 스택

- Python 3.11
- Streamlit (Frontend)
- Gemini API (LLM)
- SQLAlchemy (ORM)
- SQLite (Database)

## 주요 기능

- 사용자 로그인 및 세션 관리
- 채팅 이력 저장 및 불러오기 (최대 100개)
- LLM 기반 SQL 자동 생성
- DB 질의 결과 기반 답변 제공
- RAG 구조를 활용한 대화 흐름 유지
- RAG를 통한 답변을 구할 수 없다면 Gemini 기본 답변 제공

## 설치 및 실행 방법

1. 패키지 설치
```bash
pip install -r requirements.txt
```
2. 실행
```bash
streamlit run main.py
```


## 향후 개선 방향
- OLLAMA 또는 로컬 LLM 연동으로 Gemini API 대체
- 채팅 이력 분석 기반 자동 요약 기능
- 실제 회원 시스템 기반 DB 연동
- 사용자 맞춤형 답변 조정 기능 추가
