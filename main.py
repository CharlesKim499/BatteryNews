import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from dotenv import load_dotenv
import glob
import hmac

st.title("나만의 챗GPT💬 ")


def check_password():
    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        username = st.session_state["username"]
        password = st.session_state["password"]

        # # 디버깅을 위해 변수 값 출력
        # st.write(f"Entered Username: {username}")
        # st.write(f"Entered Password: {password}")

        # # st.secrets의 내용을 출력하여 확인
        # st.write(f"Secrets: {st.secrets}")

        if username in st.secrets["users"]:
            stored_password = st.secrets["users"][username]
            # st.write(f"Stored Password: {stored_password}")  # 디버깅을 위해 출력

            if isinstance(password, str) and isinstance(stored_password, str):
                if hmac.compare_digest(password, stored_password):
                    st.session_state["password_correct"] = True
                    del st.session_state["password"]
                    del st.session_state["username"]
                else:
                    st.session_state["password_correct"] = False
            else:
                st.session_state["password_correct"] = False
        else:
            st.session_state["password_correct"] = False

        # 디버깅을 위해 결과 출력
        # st.write(f"Password Correct: {st.session_state['password_correct']}")

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")
    return False


if not check_password():
    st.stop()

else:
    # 처음 1번만 실행하기 위한 코드
    if "messages" not in st.session_state:
        # 대화 기록을 저장하기 위한 용도로 생성한다.
        st.session_state["messages"] = []

    # 사이드바 생성
    with st.sidebar:
        # 초기화 버튼 생성
        clear_btn = st.button("대화 초기화")

        prompt_files = glob.glob("prompts/*.yaml")

        selected_prompt = st.selectbox(
            "프롬프트를 선택해 주세요", prompt_files, index=0
        )
        # selected_prompt = "prompts/battery_rag.yaml"

        # task_input = st.text_input("Task 입력", "")

    # 이전 대화를 출력
    def print_messages():
        for chat_message in st.session_state["messages"]:
            st.chat_message(chat_message.role).write(chat_message.content)

    # 새로운 메시지를 추가
    def add_message(role, message):
        st.session_state["messages"].append(ChatMessage(role=role, content=message))

    def create_chain(prompt_filepath, task=""):
        prompt = load_prompt(prompt_filepath, encoding="utf-8")
        if task:
            prompt = prompt.partial(task=task)

        # 언어 모델
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

        # 출력 파서
        outputp_parser = StrOutputParser()

        # 체인 생성
        chain = prompt | llm | outputp_parser
        return chain

    # 초기화 버튼이 눌리면 :
    if clear_btn:
        # 대화 초기화
        st.session_state["messages"] = []

    # 이전 대화 출력
    print_messages()

    # 사용자의 입력
    user_input = st.chat_input("궁금한 내용을 물어보세요.")

    # 만약 사용자 입력이 들어오면...
    if user_input:
        # 사용자의 입력
        st.chat_message("user").write(user_input)
        # chain 생성
        # chain = create_chain(selected_prompt, task=task_input)
        chain = create_chain(selected_prompt)

        # 스트리밍 호출
        response = chain.stream({"question": user_input})

        # AI의 답변
        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            containner = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                containner.markdown(ai_answer)

        # 대화 기록에 저장한다.
        add_message("user", user_input)
        add_message("assistant", ai_answer)
