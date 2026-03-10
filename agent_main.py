from langchain_community.llms import Ollama
from langchain.agents import Tool, initialize_agent, AgentType

from agent_tools import (
    policy_tool,
    faculty_workload_tool,
    dept_summary_tool,
    timetable_tool,
    free_faculty_tool
)

llm = Ollama(model="mistral")

tools = [

    Tool(
        name="PolicyRules",
        func=policy_tool,
        description="""
        Use this tool to answer questions about:
        - Maximum workload per professor
        - Teaching hour limits
        - Scheduling rules
        - University workload policies
        """
    ),

    Tool(
        name="FacultyWorkload",
        func=faculty_workload_tool,
        description="""
        Use this tool ONLY when asking about
        a specific professor's workload.
        The tool already returns the final formatted answer.
        Do NOT modify the output.
        Example:
        - What is Prof. Sharma's workload?
        - How many hours does Prof. Mehta teach?
        """
    ),

    Tool(
        name="DepartmentSummary",
        func=dept_summary_tool,
        description="""
        Use this tool when asking about
        total workload of a department.
        The tool already returns the final formatted answer.
        Do NOT rephrase or generate additional explanation.
        Example:
        - Summarize CSE department workload
        """
    ),

    Tool(
        name="TimetableLookup",
        func=timetable_tool,
        description="Use this tool to show full timetable."
    ),

    Tool(
        name="FreeFaculty",
        func=free_faculty_tool,
        description="""
        Use this tool to find free professors.
        Input format:
        Day=Tuesday, Time=14:00-15:00
        """
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

print("\nFaculty Timetable AI Agent Ready\n")

while True:
    q = input("Ask: ")
    if q == "exit":
        break
    print(agent.run(q))