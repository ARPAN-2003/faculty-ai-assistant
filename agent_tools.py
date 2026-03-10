import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------- Load CSV Data ----------

faculty_df = pd.read_csv("faculty.csv")
timetable_df = pd.read_csv("timetable.csv")

# Normalize CSV columns (important fix)
faculty_df["Name"] = faculty_df["Name"].str.strip()
faculty_df["Department"] = faculty_df["Department"].str.strip()

timetable_df["Day"] = timetable_df["Day"].str.strip().str.lower()
timetable_df["Time"] = timetable_df["Time"].str.strip()
timetable_df["Faculty"] = timetable_df["Faculty"].str.strip()

# ---------- Load Vector DB ----------

emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma(
    persist_directory="policy_db",
    embedding_function=emb
)

retriever = vectordb.as_retriever()


# ===================================
# TOOL 1 — POLICY RAG TOOL
# ===================================

def policy_tool(query: str) -> str:
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "No policy found."
    return "\n".join([d.page_content for d in docs])


# ===================================
# TOOL 2 — FACULTY WORKLOAD TOOL
# ===================================

def faculty_workload_tool(input_text: str) -> str:
    import re

    # Extract professor name pattern like "Prof. Sharma"
    match = re.search(r"prof\.?\s+[A-Za-z]+", input_text, re.IGNORECASE)

    if not match:
        return "Faculty not found."

    name = match.group(0)

    # Normalize both sides
    faculty_df["Name_clean"] = (
        faculty_df["Name"]
        .str.lower()
        .str.replace(".", "", regex=False)
        .str.strip()
    )

    name_clean = name.lower().replace(".", "").strip()

    df = faculty_df[faculty_df["Name_clean"] == name_clean]

    if df.empty:
        return "Faculty not found."

    faculty_name = df.iloc[0]["Name"]
    total_hours = df["HoursPerWeek"].sum()

    output = []
    output.append("=================================")
    output.append(" Faculty Workload Report ")
    output.append("=================================")
    output.append(f" Professor : {faculty_name}")
    output.append("")
    output.append(" Courses Assigned:")
    output.append("")

    for _, r in df.iterrows():
        output.append(f"  • {r['Course']} ({r['HoursPerWeek']} hrs/week)")

    output.append("")
    output.append(f" Total Weekly Load : {total_hours} hours")
    output.append("=================================")

    return "\n".join(output)


# ===================================
# TOOL 3 — DEPARTMENT SUMMARY TOOL
# ===================================

def dept_summary_tool(input_text: str) -> str:
    import re

    # Extract department code like CSE, ECE, EE, ME etc.
    match = re.search(r"\b(CSE|ECE|EE|ME)\b", input_text.upper())

    if not match:
        return "Department not found."

    dept = match.group(1)

    df = faculty_df[faculty_df["Department"].str.upper() == dept]

    if df.empty:
        return "Department not found."

    total_hours = df["HoursPerWeek"].sum()
    total_courses = df.shape[0]

    output = []
    output.append("=================================")
    output.append(" Department Summary Report ")
    output.append("=================================")
    output.append(f" Department : {dept}")
    output.append(f" Total Courses : {total_courses}")
    output.append(f" Total Teaching Hours : {total_hours} hours")
    output.append("")
    output.append(" Faculty-wise Distribution:")
    output.append("")

    for _, r in df.iterrows():
        output.append(
            f"  • {r['Name']} → {r['Course']} ({r['HoursPerWeek']} hrs)"
        )

    output.append("=================================")

    return "\n".join(output)


# ===================================
# TOOL 4 — TIMETABLE LOOKUP TOOL
# ===================================

def timetable_tool(query: str) -> str:
    return timetable_df.to_string(index=False)


# ===================================
# TOOL 5 — FREE FACULTY FINDER (FINAL FIX)
# ===================================

def free_faculty_tool(input_text: str) -> str:
    import re
    
    # Extract day and time from string
    day_match = re.search(r"day\s*=\s*(\w+)", input_text, re.IGNORECASE)
    time_match = re.search(r"time\s*=\s*([\d:-]+)", input_text, re.IGNORECASE)

    if not day_match or not time_match:
        return "Invalid input format. Use Day=Tuesday, Time=14:00-15:00"

    day = day_match.group(1).strip().lower()
    time = time_match.group(1).strip()

    busy = timetable_df[
        (timetable_df["Day"] == day) &
        (timetable_df["Time"] == time)
    ]["Faculty"].tolist()

    all_faculty = faculty_df["Name"].tolist()

    free = [f for f in all_faculty if f not in busy]

    if not free:
        return f"No faculty are free on {day.title()}, {time}."

    return f"Free faculty on {day.title()}, {time}:\n" + "\n".join(free)