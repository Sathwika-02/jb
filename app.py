"""Mock Interview Coach: practice software engineering interviews by voice.

Claude asks questions grounded in your resume, you answer out loud, Whisper
transcribes locally, and Claude grades the answer and shows a model answer.
"""

import datetime
import json
import os
import tempfile
from pathlib import Path

import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from pypdf import PdfReader

load_dotenv()

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
WHISPER_SIZE = os.getenv("WHISPER_MODEL", "base.en")
SESSIONS_DIR = Path(__file__).parent / "sessions"

INTERVIEW_TYPES = {
    "Resume deep-dive": "a probing question about a specific project, decision, or technology on the candidate's resume",
    "Behavioral": "a behavioral question (conflict, ownership, failure, leadership, deadlines) best answered with STAR",
    "DSA / Coding": "a data structures & algorithms problem the candidate should talk through: approach, edge cases, complexity",
    "System design": "a system design question scoped for a spoken answer (requirements, components, data model, scaling, trade-offs)",
    "CS fundamentals": "a CS fundamentals question (OS, databases, networking, concurrency, OOP)",
}


# ---------- helpers ----------

@st.cache_resource(show_spinner="Loading speech-to-text model (first run downloads it)...")
def get_whisper() -> WhisperModel:
    return WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")


@st.cache_resource
def get_claude() -> Anthropic:
    return Anthropic()


def read_resume(uploaded) -> str:
    if uploaded.name.lower().endswith(".pdf"):
        return "\n".join(page.extract_text() or "" for page in PdfReader(uploaded).pages)
    return uploaded.read().decode("utf-8", errors="ignore")


def transcribe(audio_bytes: bytes) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        tmp.write(audio_bytes)
        tmp.close()
        segments, _ = get_whisper().transcribe(tmp.name, vad_filter=True)
        return " ".join(seg.text.strip() for seg in segments).strip()
    finally:
        os.unlink(tmp.name)


def ask_claude(system: str, prompt: str, max_tokens: int = 2000) -> str:
    resp = get_claude().messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def candidate_context() -> str:
    s = st.session_state
    return (
        f"Target role: {s.role or 'Software Engineer'} ({s.level})\n\n"
        f"<resume>\n{s.resume or '(no resume provided)'}\n</resume>"
    )


def save_record(record: dict) -> None:
    SESSIONS_DIR.mkdir(exist_ok=True)
    path = SESSIONS_DIR / f"{datetime.date.today().isoformat()}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def voice_or_text(label: str, key: str) -> str:
    """Let the user answer by microphone or by typing; returns the text."""
    audio = st.audio_input(f"🎤 {label} (record)", key=f"{key}_audio")
    typed = st.text_area(f"…or type it", key=f"{key}_text", height=120)
    if audio is not None:
        cache_key = f"{key}_transcript"
        audio_bytes = audio.getvalue()
        if st.session_state.get(f"{cache_key}_src") != hash(audio_bytes):
            with st.spinner("Transcribing..."):
                st.session_state[cache_key] = transcribe(audio_bytes)
                st.session_state[f"{cache_key}_src"] = hash(audio_bytes)
        text = st.session_state[cache_key]
        st.markdown(f"**Transcript:** {text or '_(nothing detected — try again)_'}")
        return text
    return typed.strip()


# ---------- UI ----------

st.set_page_config(page_title="Mock Interview Coach", page_icon="🎯", layout="wide")
st.session_state.setdefault("resume", "")
st.session_state.setdefault("question", "")
st.session_state.setdefault("asked", [])

with st.sidebar:
    st.header("Your profile")
    uploaded = st.file_uploader("Resume (PDF or TXT)", type=["pdf", "txt", "md"])
    if uploaded is not None:
        st.session_state.resume = read_resume(uploaded)
        st.success(f"Resume loaded ({len(st.session_state.resume.split())} words)")
    st.text_input("Target role", key="role", placeholder="e.g. Backend Engineer, Java/Spring")
    st.selectbox("Level", ["Intern / New grad", "Mid-level (2-5 yrs)", "Senior (5+ yrs)"], key="level", index=0)
    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error("ANTHROPIC_API_KEY not set — add it to a .env file.")

st.title("🎯 Mock Interview Coach")
mock_tab, study_tab, history_tab = st.tabs(["Mock interview", "Study a question", "History"])

SYSTEM_INTERVIEWER = (
    "You are a senior software engineering interviewer at a top tech company. "
    "You tailor questions to the candidate's resume and target level."
)
SYSTEM_COACH = (
    "You are an expert software engineering interview coach. Be direct and specific. "
    "When writing model answers, ground them in the candidate's real resume — never invent "
    "projects, employers, or metrics they don't have; if something is missing, say what they "
    "should fill in. Use Markdown."
)

# --- Mock interview: Claude asks, you answer by voice, Claude grades ---
with mock_tab:
    qtype = st.selectbox("Round type", list(INTERVIEW_TYPES))
    if st.button("Ask me a question", type="primary"):
        with st.spinner("Thinking of a question..."):
            previous = "\n".join(f"- {q}" for q in st.session_state.asked[-15:]) or "(none)"
            st.session_state.question = ask_claude(
                SYSTEM_INTERVIEWER,
                f"{candidate_context()}\n\nAsk ONE question: {INTERVIEW_TYPES[qtype]}.\n"
                f"Don't repeat these:\n{previous}\n\nOutput only the question, as you'd say it aloud.",
                max_tokens=400,
            ).strip()
            st.session_state.asked.append(st.session_state.question)

    if st.session_state.question:
        st.info(f"**Interviewer:** {st.session_state.question}")
        answer = voice_or_text("Your answer", key="mock")
        if st.button("Get feedback", disabled=not answer):
            with st.spinner("Evaluating..."):
                feedback = ask_claude(
                    SYSTEM_COACH,
                    f"{candidate_context()}\n\nRound: {qtype}\n"
                    f"Question: {st.session_state.question}\n\n"
                    f"<candidate_answer>\n{answer}\n</candidate_answer>\n\n"
                    "Give:\n1. **Score** out of 10 with a one-line verdict\n"
                    "2. **What worked**\n3. **Gaps / mistakes** (technical accuracy, structure, missing points)\n"
                    "4. **Delivery** (rambling, filler, clarity) based on the transcript\n"
                    "5. **Model answer** — a strong spoken answer in the candidate's voice, grounded in their resume\n"
                    "6. **Likely follow-up questions** (2-3)",
                )
            st.markdown(feedback)
            save_record({
                "time": datetime.datetime.now().isoformat(timespec="seconds"),
                "mode": "mock", "type": qtype,
                "question": st.session_state.question, "answer": answer, "feedback": feedback,
            })

# --- Study: say or type a question you expect, get a prep sheet ---
with study_tab:
    st.caption("Speak or type a question you expect to be asked; get a prep sheet to study from.")
    question = voice_or_text("Question", key="study")
    if st.button("Build prep sheet", disabled=not question):
        with st.spinner("Preparing..."):
            sheet = ask_claude(
                SYSTEM_COACH,
                f"{candidate_context()}\n\nQuestion to prepare: {question}\n\n"
                "Give:\n1. **What the interviewer is really testing**\n"
                "2. **Key points to hit** (bullets)\n"
                "3. **Model answer** grounded in the resume (for coding: approach, pseudocode/code, complexity)\n"
                "4. **Common mistakes**\n5. **Likely follow-ups** with brief answers",
                max_tokens=3000,
            )
        st.markdown(sheet)
        save_record({
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "mode": "study", "question": question, "feedback": sheet,
        })

# --- History ---
with history_tab:
    files = sorted(SESSIONS_DIR.glob("*.jsonl"), reverse=True) if SESSIONS_DIR.exists() else []
    if not files:
        st.write("No sessions yet.")
    for path in files:
        st.subheader(path.stem)
        for line in reversed(path.read_text(encoding="utf-8").splitlines()):
            rec = json.loads(line)
            with st.expander(f"{rec['time'][11:]} · {rec['mode']} · {rec['question'][:90]}"):
                if rec.get("answer"):
                    st.markdown(f"**Your answer:** {rec['answer']}")
                st.markdown(rec["feedback"])
