import os

if not os.path.exists("policy_db"):
    import store_rules

import streamlit as st
from langchain_huggingface import HuggingFaceHub
from langchain.agents import Tool, initialize_agent, AgentType

from agent_tools import (
    policy_tool,
    faculty_workload_tool,
    dept_summary_tool,
    timetable_tool,
    free_faculty_tool
)

# ---------------------------
# PAGE CONFIG
# ---------------------------

st.set_page_config(
    page_title="Faculty AI Assistant",
    page_icon="🎓",
    layout="wide"
)

# ---------------------------
# DARK THEME STYLING
# ---------------------------

st.markdown("""
<style>
body {
    background-color: #0E1117;
}

.big-title {
    font-size: 36px;
    font-weight: 700;
    color: #00F5FF;
}

.hero-title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    color: #00F5FF;
    margin-top: 30px;
    margin-bottom: 10px;
}

.hero-subtitle {
    text-align: center;
    font-size: 18px;
    color: #AAAAAA;
    margin-bottom: 40px;
}

.footer {
    text-align: center;
    color: gray;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# HEADER
# ---------------------------

st.markdown('<div class="hero-title">🎓 Faculty Workload & Timetable AI Assistant</div>', unsafe_allow_html=True)

st.markdown('<div class="hero-subtitle">Ask anything about faculty workload, timetable, or university policies.</div>', unsafe_allow_html=True)

st.divider()

# ---------------------------
# LOAD AGENT (CACHED)
# ---------------------------

@st.cache_resource
def load_agent():
    llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    model_kwargs={"temperature":0.5, "max_length":512}
)

    tools = [
        Tool(
            name="PolicyRules",
            func=policy_tool,
            description="Answer questions about workload policies."
        ),
        Tool(
            name="FacultyWorkload",
            func=faculty_workload_tool,
            description="Get workload of a specific professor."
        ),
        Tool(
            name="DepartmentSummary",
            func=dept_summary_tool,
            description="Summarize department workload."
        ),
        Tool(
            name="TimetableLookup",
            func=timetable_tool,
            description="Show full timetable."
        ),
        Tool(
            name="FreeFaculty",
            func=free_faculty_tool,
            description="Find free faculty. Format: Day=Tuesday, Time=14:00-15:00"
        ),
    ]

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=2,
        early_stopping_method="force"
    )

    return agent


agent = load_agent()

# ---------------------------
# CHAT MEMORY
# ---------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------
# DISPLAY OLD MESSAGES
# ---------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------
# CHAT INPUT (NO ERRORS)
# ---------------------------

if prompt := st.chat_input("💬 Type your question here..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            response = agent.run(prompt)
            st.markdown(response)

    # Save assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

st.divider()

st.markdown('<p class="footer">Built using LangChain + Ollama + Streamlit</p>', unsafe_allow_html=True)




