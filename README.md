# Mock Interview Coach

Practice software engineering interviews out loud. Upload your resume, and Claude
asks you questions tailored to it. You answer by voice, Whisper transcribes the
answer locally, and you get a score, feedback, a model answer grounded in your
resume, and likely follow-up questions.

## Setup (Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # then put your Anthropic API key in .env
streamlit run app.py
```

The app opens at http://localhost:8501. The first recording downloads the Whisper
model (~150 MB for `base.en`).

## Features

- **Mock interview:** pick a round (resume deep-dive, behavioral, DSA, system design,
  CS fundamentals). Then answer by mic or by typing, and get graded feedback.
- **Study a question:** say or type a question you expect to be asked, and get a prep
  sheet (what the interviewer is testing, key points, model answer, pitfalls, follow-ups).
- **History:** every session is saved to `sessions/YYYY-MM-DD.jsonl` so you can review it.

Speech-to-text runs on your machine. Only the transcript text and your resume
are sent to the Claude API.

## Live Interview Assistant (for mock interviews)

`live_assistant.py` keeps listening during a call. It never stops until you close
the window. It hears what your computer plays (the other people on the call),
transcribes each sentence locally, and when someone asks a question, it streams
a short answer into the window.

```powershell
.venv\Scripts\activate
python live_assistant.py                      # or: python live_assistant.py --resume resume.pdf
```

- **Auto-answer** (on by default): every question from the other side gets an answer.
  Small talk is ignored, and the same question isn't answered twice.
- **Answer last:** forces an answer to the latest question, even a half-finished one.
- **Ask:** type any question at the bottom and press Enter.
- **Target role:** e.g. "Backend Engineer, Java/Spring, new grad". Answers are tailored to it.
- **Resume:** load it once with the Resume… button or `--resume`. It is saved to `resume.txt` and loaded automatically next time. Answers are written in first person from your real experience (STAR for behavioral questions), and nothing is made up.
- **Also hear my mic:** also transcribes your own voice, so answers can follow the
  whole conversation. Use headphones, or your mic will pick up the call twice.
- **On top:** keeps the window above the video call.
- Works with any call app (Zoom, Meet, Teams, Discord) because it records system audio.
  If you switch speakers or headphones mid-call, it follows the new device.
- Answers are also saved to `sessions/`, and they show up in the History tab of `app.py`.
- **Which LLM:** set `LLM_PROVIDER` in `.env`. `groq` and `gemini` have free tiers (keys from console.groq.com/keys and aistudio.google.com/apikey). `openai` (ChatGPT) and `anthropic` (Claude) need paid API credit.

## Demo Coach (for client product demos)

`demo_coach.py` is a presenter-only helper for live demos. It listens to the call the
same way the live assistant does, and when a client asks a question it streams a short
answer based on your product docs, with a `Show:` hint for what to demo next.

```powershell
.venv\Scripts\activate
python demo_coach.py --docs product-overview.pdf faq.md security.md
```

- **Hide from screen share** (on by default): the coach window is left out of screen
  capture, so clients see only your product. You still see it on your own monitor, like
  a presenter view. The green/red label shows whether hiding is active (requires
  Windows 10 version 2004 or later).
- **Product docs…:** load PDF/TXT/MD files. They're saved to `product_docs.txt` and
  reloaded automatically next time. Answers only use what's in the docs. When the docs
  don't cover a question, the answer starts with "Not in docs —" so nobody improvises
  pricing, features, or commitments.
- **Client:** a short note on who you're presenting to (e.g. "Acme, logistics, cares about
  SSO and audit logs"). Answers are tailored to it.
- **Auto-answer**, **Answer last**, **Ask**, and **Hear my mic** work like they do in the live assistant.
- Answers are saved to `sessions/` with mode `demo`.

## Mobile Chat (Groq, phone-friendly)

`mobile_chat.py` serves a chat page you can open on your phone. Answers come from Groq
with no system prompt, the same as the Groq Playground, and render headings, tables,
and code, with a collapsible "Reasoning" section.

```powershell
.venv\Scripts\activate
python mobile_chat.py
```

Open the "On phone" link it prints (your phone must be on the same Wi-Fi). The link has a
private `?key=`, so other people on the network can't use your API key. If Windows Firewall
asks the first time, allow access on **Private networks**. To add a system prompt, set
`CHAT_SYSTEM=...` in `.env`.
