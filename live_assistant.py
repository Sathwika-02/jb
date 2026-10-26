"""Live Call Assistant: listens to a call continuously and answers questions as they're asked.

It captures what your computer plays (the other people on the call) and,
optionally, your microphone. Whisper transcribes each utterance locally, and
an LLM (ChatGPT or Claude) decides whether it was a question and streams an answer.
It keeps running until you close the window.

    python live_assistant.py [--resume path/to/resume.pdf]
"""

import argparse
import collections
import contextlib
import datetime
import io
import json
import os
import queue
import threading
import time
import tkinter as tk
import warnings
import wave
from pathlib import Path
from tkinter import filedialog, scrolledtext

import numpy as np
import soundcard as sc
from dotenv import load_dotenv
from faster_whisper import WhisperModel

load_dotenv()
warnings.filterwarnings("ignore", category=sc.SoundcardRuntimeWarning)

# LLM_PROVIDER: openai (ChatGPT, default), groq / gemini (free tiers), or anthropic (Claude).
# Groq and Gemini speak the OpenAI API, so they share the OpenAI code path with a different URL.
#   provider: (display name, API key variable, base URL, model variable, default model)
OPENAI_COMPATIBLE = {
    "openai": ("ChatGPT", "OPENAI_API_KEY", None, "OPENAI_MODEL", "gpt-4.1"),
    "groq": ("Groq", "GROQ_API_KEY", "https://api.groq.com/openai/v1",
             "GROQ_MODEL", "openai/gpt-oss-120b"),
    "gemini": ("Gemini", "GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai/",
               "GEMINI_MODEL", "gemini-2.5-flash"),
}
# Fallback order if the configured model gets retired (substring match on model ids).
PREFERRED_MODELS = ("gpt-oss-120b", "gpt-4.1", "gemini-2.5-flash", "llama-3.3-70b", "qwen", "gpt-oss-20b")
PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
if PROVIDER in OPENAI_COMPATIBLE:
    PROVIDER_NAME, API_KEY_VAR, BASE_URL, model_var, default_model = OPENAI_COMPATIBLE[PROVIDER]
    MODEL = os.getenv("LIVE_MODEL") or os.getenv(model_var, default_model)
else:
    PROVIDER_NAME, API_KEY_VAR = "Claude", "ANTHROPIC_API_KEY"
    MODEL = os.getenv("LIVE_MODEL") or os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
WHISPER_SIZE = os.getenv("WHISPER_MODEL", "base.en")
# Smaller, faster model for the grey live text shown while someone is still talking.
PARTIAL_WHISPER_SIZE = os.getenv("PARTIAL_WHISPER_MODEL", "tiny.en")
SESSIONS_DIR = Path(__file__).parent / "sessions"
SAVED_RESUME = Path(__file__).parent / "resume.txt"   # last loaded resume, reused on next start

RATE = 16000
BLOCK = RATE // 10            # 100 ms per read
PRE_ROLL_BLOCKS = 5           # keep 500 ms before speech starts so soft openings ("Hello") aren't clipped
SILENCE_END_BLOCKS = 6        # 600 ms of quiet ends an utterance
PARTIAL_EVERY_BLOCKS = 5      # refresh live text every 500 ms while speaking
MIN_LEVEL = float(os.getenv("MIN_SPEECH_LEVEL", "0.0025"))  # RMS floor that counts as speech
# Whisper defaults to 4 CPU threads; using most cores roughly halves transcription time.
FINAL_THREADS = max(4, (os.cpu_count() or 4) - 4)
# Context hint so short greetings and tech terms are heard right ("Hello hi", not "Okay bye";
# "Dijkstra's", "BFS"). Add your own project/company names via WHISPER_PROMPT in .env.
WHISPER_PROMPT = os.getenv("WHISPER_PROMPT",
                           "Hello, hi. Technical interview about software engineering. Terms: "
                           "Dijkstra's algorithm, BFS, DFS, binary tree, linked list, hash map, dynamic "
                           "programming, Big O, time complexity, rate limiter, load balancer, Redis, Kafka, "
                           "Kubernetes, Docker, AWS, Lambda, DynamoDB, PostgreSQL, MongoDB, React, Next.js, "
                           "Node.js, Express, GraphQL, REST API, WebSocket, WebRTC, microservices, "
                           "TypeScript, CI/CD.")
# Final transcripts go to Groq's hosted Whisper large-v3 when a Groq key is set: ~0.4 s per
# utterance and far more accurate (accents, tech terms) than the local model, which stays as
# the fallback and still drives the grey live text. STT_PROVIDER=local forces local only.
STT_PROVIDER = os.getenv("STT_PROVIDER", "groq" if os.getenv("GROQ_API_KEY") else "local").lower()
GROQ_STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3")
CLOUD_COOLDOWN = 60           # after a cloud error (e.g. free-tier rate limit), use local for 60 s
# Phrases Whisper tends to invent from noise; dropped when they're the whole utterance.
HALLUCINATIONS = {"thank you.", "thank you", "thanks for watching!", "thanks for watching.", "you", "bye."}
MIN_SPEECH_BLOCKS = 4         # ignore blips shorter than 400 ms
MAX_UTTERANCE_BLOCKS = 200    # split monologues every 20 s
CONTEXT_LINES = 30            # transcript lines sent to Claude

SYSTEM = """You are a real-time interview assistant. The user is a candidate in a live mock
interview on a video call; friends are playing the interviewers. You get a rolling transcript:
lines tagged [Them] are the interviewers, [You] is the candidate. Transcripts come from speech
recognition, so expect missing punctuation and misheard words (e.g. "Jango" = Django) —
infer the intended meaning.

When an interviewer asks a question, write the answer exactly as the candidate would say it out
loud: natural, conversational spoken English, first person ("So basically...", "In my project,
I..."). It should sound like a person talking, not a textbook or a slide.

Answer format:
- First line: `Q: <the question, cleaned up>`
- Then the spoken answer in 1-3 short paragraphs of plain sentences. No bullet points, no
  headings, no bold/italics, no Markdown symbols at all (the screen shows plain text).
- Start with a simple one-line explanation, as if explaining to a friend.
- Always include a concrete, real-world example ("For example, when you open Swiggy and it
  loads your past orders, the app is calling a REST API like GET /orders..."). Prefer an
  example from the candidate's resume if one fits ("In my X project, I used this to...").
- By question type:
  - About me / resume / projects: talk through the most relevant project from the resume,
    what I did, the tech used, and the result.
  - Behavioral ("tell me about a time..."): tell it as a short story (situation, what I did,
    how it turned out), using a real experience from the resume.
  - Technical concepts: simple explanation, how it works in plain words, an example, and
    one trade-off or "when you'd use it".
  - Coding / DSA: explain the approach the way you'd talk it through on a call, then the
    time and space complexity; code only if asked.
  - System design: walk through it conversationally: what it needs to do, the main pieces,
    how data flows, how it scales.
  - "Why this role/company", strengths, weaknesses: honest, tied to the resume and target role.
- Keep it to about 80-150 words so it can be said in under a minute.

Never invent employers, colleges, projects, metrics, or skills that aren't on the resume, and
never fill gaps with placeholders like "XYZ University" — just leave that detail out. If the resume
has nothing relevant, give a general answer and end with a line like "(no matching resume
experience — speak to how you'd approach it)"."""

AUTO_INSTRUCTION = """Look at the most recent [Them] lines. If they contain a question or request
that the user would need to answer, and it isn't one already listed under <already_answered>,
answer it in the format above. Otherwise (small talk, statements, incomplete thought, already
answered) output exactly: NONE"""

FORCE_INSTRUCTION = """Answer the most recent question asked in the transcript (from anyone),
in the format above, even if it's partial — make your best guess at what is being asked."""


# ---------- audio ----------

class Listener(threading.Thread):
    """Records one audio source forever and emits (label, audio, utterance id) utterances.

    While someone is still talking it also offers the audio so far to `partials`,
    so the UI can show live text before the utterance ends."""

    def __init__(self, label: str, loopback: bool, out: queue.Queue, ui: queue.Queue,
                 partials: queue.Queue | None = None):
        super().__init__(daemon=True)
        self.label, self.loopback, self.out, self.ui = label, loopback, out, ui
        self.partials = partials
        self.enabled = threading.Event()
        self.level = 0.0
        self.utt = 0

    def devices(self) -> list:
        """Loopback records every output device, not just the default: call apps (Teams, Meet)
        often play through the 'communications' device, and Bluetooth headsets switch to a
        separate Hands-Free device during calls."""
        if self.loopback:
            return [sc.get_microphone(s.id, include_loopback=True) for s in sc.all_speakers()]
        return [sc.default_microphone()]

    def run(self):
        while True:
            if not self.enabled.wait(timeout=0.5):
                continue
            try:
                self.record_until_device_changes()
            except Exception as e:  # device unplugged, driver hiccup, ...
                self.ui.put(("status", f"{self.label} audio error: {e} — retrying"))
                time.sleep(1)

    def record_until_device_changes(self):
        mics = self.devices()
        # Noise floor = a quiet percentile of the last 5 s, so steady background hiss on the call
        # can't keep an utterance "open" (which delayed transcripts by many seconds).
        recent = collections.deque([0.002] * 50, maxlen=50)
        pre, speech, silent = [], [], 0
        last_check = time.time()
        with contextlib.ExitStack() as stack:
            recs, names = [], []
            for mic in mics:
                try:  # an endpoint that's present but unusable shouldn't block the others
                    recs.append(stack.enter_context(mic.recorder(samplerate=RATE, channels=1, blocksize=BLOCK)))
                    names.append(mic.name)
                except Exception:
                    pass
            if not recs:
                raise RuntimeError("no audio device could be opened")
            self.ui.put(("status", f"Listening to {self.label}: {', '.join(names)}"))
            while self.enabled.is_set():
                # Mix all devices; whichever one the call is playing on carries the voice.
                block = sum(rec.record(numframes=BLOCK)[:, 0] for rec in recs).astype(np.float32)
                rms = float(np.sqrt(np.mean(block ** 2))) if block.size else 0.0
                self.level = rms
                recent.append(rms)
                noise = float(np.percentile(recent, 10))
                # Hysteresis: a clear voice starts an utterance, but quieter syllables keep it going,
                # so soft word endings on a noisy call don't cut the sentence short.
                loud = (rms > max(MIN_LEVEL * 0.6, noise * 1.8) if speech
                        else rms > max(MIN_LEVEL, noise * 3))

                if speech:
                    speech.append(block)
                    silent = 0 if loud else silent + 1
                    if silent >= SILENCE_END_BLOCKS or len(speech) >= MAX_UTTERANCE_BLOCKS:
                        # None audio still tells the UI to drop this utterance's live text.
                        keep = len(speech) - silent >= MIN_SPEECH_BLOCKS
                        self.out.put((self.label, np.concatenate(speech) if keep else None, self.utt))
                        speech, silent = [], 0
                    elif self.partials is not None and len(speech) % PARTIAL_EVERY_BLOCKS == 0:
                        try:
                            self.partials.put_nowait((self.label, self.utt, np.concatenate(speech)))
                        except queue.Full:
                            pass  # live text is still busy; the next refresh will catch up
                elif loud:
                    self.utt += 1
                    speech = pre + [block]
                    pre = []
                else:
                    pre = (pre + [block])[-PRE_ROLL_BLOCKS:]

                # Pick up devices added or removed mid-call (e.g. Bluetooth headphones connecting).
                if time.time() - last_check > 3 and not speech:
                    last_check = time.time()
                    if [m.id for m in self.devices()] != [m.id for m in mics]:
                        return


class Transcriber(threading.Thread):
    def __init__(self, audio: queue.Queue, ui: queue.Queue, on_text):
        super().__init__(daemon=True)
        self.audio, self.ui, self.on_text = audio, ui, on_text

    def run(self):
        self.cloud, self.cloud_off_until = None, 0.0
        if STT_PROVIDER == "groq":
            from openai import OpenAI
            self.cloud = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
        self.ui.put(("status", f"Loading Whisper '{WHISPER_SIZE}' (first run downloads it)..."))
        model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8", cpu_threads=FINAL_THREADS)
        self.ui.put(("status", "Listening..." + (f" (speech-to-text: Groq {GROQ_STT_MODEL})" if self.cloud else "")))
        while True:
            label, audio, utt = self.audio.get()
            text = ""
            if audio is not None:
                text = self.transcribe_cloud(audio)
                if text is None:
                    try:
                        text = transcribe(model, audio)
                    except Exception as e:
                        self.ui.put(("status", f"Transcription error: {e}"))
                        text = ""
            self.ui.put(("finalized", (label, utt)))
            if any(c.isalnum() for c in text) and text.strip().lower() not in HALLUCINATIONS:
                self.on_text(label, text)

    def transcribe_cloud(self, audio) -> str | None:
        """Groq Whisper large-v3; None means "use the local model instead"."""
        if self.cloud is None or time.time() < self.cloud_off_until:
            return None
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        try:
            result = self.cloud.audio.transcriptions.create(
                model=GROQ_STT_MODEL, file=("speech.wav", buf.getvalue()), language="en",
                temperature=0, prompt=WHISPER_PROMPT or None, timeout=8)
            return result.text.strip()
        except Exception as e:
            self.cloud_off_until = time.time() + CLOUD_COOLDOWN
            self.ui.put(("status", f"Groq speech-to-text failed ({e.__class__.__name__}) — using local for 60 s"))
            return None


class PartialTranscriber(threading.Thread):
    """Quick, rough transcription of speech that's still in progress (shown greyed out)."""

    def __init__(self, partials: queue.Queue, ui: queue.Queue, finals: queue.Queue):
        super().__init__(daemon=True)
        self.partials, self.ui, self.finals = partials, ui, finals

    def run(self):
        model = WhisperModel(PARTIAL_WHISPER_SIZE, device="cpu", compute_type="int8", cpu_threads=2)
        while True:
            label, utt, audio = self.partials.get()
            if not self.finals.empty():
                continue  # a finished utterance is waiting; don't steal CPU from it
            try:
                text = transcribe(model, audio)
            except Exception:
                continue
            if any(c.isalnum() for c in text):
                self.ui.put(("partial", (label, utt, text)))


def transcribe(model: WhisperModel, audio) -> str:
    # Audio is already speech-gated by Listener, so the VAD only needs to trim silence; the
    # default threshold (0.5) drops quiet laptop-mic speech entirely.
    segments, _ = model.transcribe(audio, language="en", beam_size=1, vad_filter=True,
                                   vad_parameters={"threshold": 0.25, "min_silence_duration_ms": 1000},
                                   initial_prompt=WHISPER_PROMPT or None,
                                   without_timestamps=True, condition_on_previous_text=False)
    return " ".join(s.text.strip() for s in segments).strip()


# ---------- Claude ----------

class Answerer(threading.Thread):
    """Turns transcript updates into streamed answers. Collapses backlog so it never lags far behind."""

    system = SYSTEM   # subclasses (e.g. demo_coach.py) swap in their own prompt
    mode = "live"     # tag for records in sessions/

    def __init__(self, ui: queue.Queue, app: "App"):
        super().__init__(daemon=True)
        self.ui, self.app = ui, app
        self.jobs: queue.Queue = queue.Queue()
        self.client = None  # created on first use so a missing API key doesn't crash startup
        self.model = MODEL

    def stream_text(self, prompt: str):
        """Yield the answer text as it streams in, from whichever provider is configured."""
        if self.client is None:
            if PROVIDER in OPENAI_COMPATIBLE:
                from openai import OpenAI
                self.client = OpenAI(api_key=os.getenv(API_KEY_VAR), base_url=BASE_URL)
            else:
                from anthropic import Anthropic
                self.client = Anthropic()
        if PROVIDER in OPENAI_COMPATIBLE:
            # OpenAI's newer models only accept max_completion_tokens; Groq/Gemini use max_tokens.
            limit = {"max_completion_tokens" if PROVIDER == "openai" else "max_tokens": 4000}
            resp = self.client.chat.completions.create(
                model=self.model, stream=True, **limit,
                messages=[{"role": "system", "content": self.system}, {"role": "user", "content": prompt}],
            )
            try:
                for event in resp:
                    if event.choices and event.choices[0].delta.content:
                        yield event.choices[0].delta.content
            finally:
                resp.close()
        else:
            with self.client.messages.stream(
                model=self.model, max_tokens=1500, system=self.system,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                yield from stream.text_stream

    def switch_model(self) -> bool:
        """The configured model was retired: pick the best chat model the account still has."""
        if PROVIDER not in OPENAI_COMPATIBLE:
            return False
        try:
            ids = [m.id for m in self.client.models.list().data]
        except Exception:
            return False
        skip = ("whisper", "guard", "tts", "orpheus", "embed", "vision", "image", "audio")
        chat = [i for i in ids if i != self.model and not any(s in i.lower() for s in skip)]
        preferred = [i for p in PREFERRED_MODELS for i in chat if p in i.lower()]
        if not (preferred or chat):
            return False
        old, self.model = self.model, (preferred or chat)[0]
        self.ui.put(("status", f"Model '{old}' unavailable — switched to '{self.model}'"))
        return True

    def run(self):
        while True:
            jobs = [self.jobs.get()]
            while not self.jobs.empty():
                jobs.append(self.jobs.get_nowait())
            manual = [text for kind, text in jobs if kind == "manual"]
            for text in manual:
                self.answer(f"Answer this question the user typed: {text}", allow_none=False)
            if any(kind == "force" for kind, _ in jobs):
                self.answer(FORCE_INSTRUCTION, allow_none=False)
            elif any(kind == "auto" for kind, _ in jobs):
                self.answer(AUTO_INSTRUCTION, allow_none=True)

    def context_blocks(self) -> str:
        """Background material placed before the transcript in every prompt."""
        app = self.app
        role = app.context.get().strip()
        return ((f"<resume>\n{app.resume}\n</resume>\n\n" if app.resume else "")
                + (f"<target_role>\n{role}\n</target_role>\n\n" if role else ""))

    def answer(self, instruction: str, allow_none: bool):
        app = self.app
        with app.lock:
            transcript = "\n".join(app.transcript[-CONTEXT_LINES:]) or "(nothing yet)"
            answered = "\n".join(f"- {q}" for q in app.answered[-20:]) or "(none)"
        prompt = (
            self.context_blocks()
            + f"<transcript>\n{transcript}\n</transcript>\n\n"
            + f"<already_answered>\n{answered}\n</already_answered>\n\n{instruction}"
        )
        self.ui.put(("status", "Thinking..."))
        try:
            buf, started, full = "", False, ""
            stream = self.stream_text(prompt)
            for chunk in stream:
                full += chunk
                if started:
                    self.ui.put(("answer", chunk))
                    continue
                buf += chunk
                head = buf.lstrip()
                if allow_none and head.startswith("NONE"):
                    break
                if allow_none and "NONE".startswith(head):
                    continue  # could still turn out to be NONE; wait for more
                started = True
                self.ui.put(("answer_start", None))
                self.ui.put(("answer", buf))
            stream.close()
        except Exception as e:
            if not started and getattr(e, "status_code", None) == 404 and self.switch_model():
                return self.answer(instruction, allow_none)
            self.ui.put(("status", f"{PROVIDER_NAME} error: {e}"))
            return
        if not started:
            self.ui.put(("status", "Listening..."))
            return
        self.ui.put(("answer_end", None))
        self.ui.put(("status", "Listening..."))
        question = full.strip().splitlines()[0].removeprefix("Q:").strip()
        with app.lock:
            app.answered.append(question)
        save_record(question, full.strip(), self.mode)


def save_record(question: str, answer: str, mode: str = "live") -> None:
    """Same format as app.py, so live answers show up in its History tab."""
    SESSIONS_DIR.mkdir(exist_ok=True)
    path = SESSIONS_DIR / f"{datetime.date.today().isoformat()}.jsonl"
    record = {
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "mode": mode, "question": question, "feedback": answer,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_resume(path: str) -> str:
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    else:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    SAVED_RESUME.write_text(text, encoding="utf-8")
    return text


# ---------- UI ----------

class App:
    def __init__(self, root: tk.Tk, resume_path: str | None):
        self.root = root
        self.ui: queue.Queue = queue.Queue()
        self.audio: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.transcript: list[str] = []
        self.answered: list[str] = []
        if resume_path:
            self.resume = read_resume(resume_path)
        elif SAVED_RESUME.exists():
            self.resume = SAVED_RESUME.read_text(encoding="utf-8")
        else:
            self.resume = ""

        root.title("Live Interview Assistant")
        root.geometry("620x760")

        bar = tk.Frame(root)
        bar.pack(fill="x", padx=8, pady=(8, 4))
        self.auto = tk.BooleanVar(value=True)
        self.use_mic = tk.BooleanVar(value=False)
        self.on_top = tk.BooleanVar(value=False)
        tk.Checkbutton(bar, text="Auto-answer", variable=self.auto).pack(side="left")
        tk.Checkbutton(bar, text="Also hear my mic", variable=self.use_mic,
                       command=self.toggle_mic).pack(side="left")
        tk.Checkbutton(bar, text="On top", variable=self.on_top,
                       command=lambda: root.attributes("-topmost", self.on_top.get())).pack(side="left")
        tk.Button(bar, text="Clear", command=self.clear).pack(side="right")
        tk.Button(bar, text="Resume…", command=self.pick_resume).pack(side="right", padx=4)
        tk.Button(bar, text="Answer last", command=self.answer_last).pack(side="right")

        ctx = tk.Frame(root)
        ctx.pack(fill="x", padx=8)
        tk.Label(ctx, text="Target role:").pack(side="left")
        self.context = tk.StringVar()
        tk.Entry(ctx, textvariable=self.context).pack(side="left", fill="x", expand=True)

        ask = tk.Frame(root)
        ask.pack(fill="x", padx=8, pady=(4, 0))
        tk.Label(ask, text="Ask:").pack(side="left")
        self.ask = tk.Entry(ask, font=("Segoe UI", 11))
        self.ask.pack(side="left", fill="x", expand=True)
        self.ask.bind("<Return>", self.submit_typed)
        tk.Button(ask, text="Send", command=self.submit_typed).pack(side="left", padx=(4, 0))
        tk.Button(ask, text="Fix last", command=self.edit_last).pack(side="left", padx=(4, 0))

        status = tk.Frame(root)
        status.pack(fill="x", padx=8, pady=4)
        self.status = tk.Label(status, text="Starting...", anchor="w", fg="#555")
        self.status.pack(side="left", fill="x", expand=True)
        self.meter = tk.Canvas(status, width=80, height=10, bg="#ddd", highlightthickness=0)
        self.meter.pack(side="right")

        panes = tk.PanedWindow(root, orient="vertical", sashwidth=6)
        panes.pack(fill="both", expand=True, padx=8)
        self.trans_box = scrolledtext.ScrolledText(panes, height=8, wrap="word", font=("Segoe UI", 9))
        self.ans_box = scrolledtext.ScrolledText(panes, wrap="word", font=("Segoe UI", 12))
        self.trans_box.tag_config("You", foreground="#2a6")
        self.trans_box.tag_config("Them", foreground="#36c")
        self.ans_box.tag_config("sep", foreground="#aaa")
        self.trans_box.tag_config("Typed", foreground="#c60")
        self.trans_box.bind("<Return>", self.submit_from_transcript)
        panes.add(self.trans_box, minsize=60)
        panes.add(self.ans_box, minsize=200)

        if not os.getenv(API_KEY_VAR):
            self.status.config(text=f"{API_KEY_VAR} not set — add it to .env", fg="red")

        self.answerer = Answerer(self.ui, self)
        self.answerer.start()
        self.start_audio()
        if self.resume:
            self.ui.put(("status", f"Resume loaded ({len(self.resume.split())} words)"))
        else:
            self.ui.put(("status", "No resume loaded — click Resume… so answers use your experience"))
        self.pump()

    def start_audio(self):
        """Start the listeners and transcribers that feed the transcript box."""
        self.partial_text: dict[str, tuple[int, str]] = {}   # label -> (utterance id, live text)
        self.finalized: dict[str, int] = {}                   # label -> last finished utterance id
        self.trans_box.tag_config("partial", foreground="#999")
        partials: queue.Queue = queue.Queue(maxsize=2)
        Transcriber(self.audio, self.ui, self.on_text).start()
        PartialTranscriber(partials, self.ui, self.audio).start()
        self.them = Listener("Them", loopback=True, out=self.audio, ui=self.ui, partials=partials)
        self.you = Listener("You", loopback=False, out=self.audio, ui=self.ui, partials=partials)
        self.them.start()
        self.you.start()
        self.them.enabled.set()

    def redraw_partials(self):
        """Live (not yet final) lines always sit at the bottom of the transcript box."""
        box = self.trans_box
        ranges = box.tag_ranges("partial")
        for start, end in reversed(list(zip(ranges[::2], ranges[1::2]))):
            box.delete(start, end)
        for label, (_, text) in self.partial_text.items():
            self.append(box, f"{label}: {text} …\n", "partial")

    # called from the transcriber thread
    def on_text(self, label: str, text: str):
        with self.lock:
            self.transcript.append(f"[{label}] {text}")
        self.ui.put(("transcript", (label, text)))
        if label == "Them" and self.auto.get():
            self.answerer.jobs.put(("auto", None))

    def toggle_mic(self):
        (self.you.enabled.set if self.use_mic.get() else self.you.enabled.clear)()

    def submit_typed(self, _event=None):
        text = self.ask.get().strip()
        if text:
            self.ask.delete(0, "end")
            self.answerer.jobs.put(("manual", text))
        self.ask.focus_set()

    def submit_from_transcript(self, _event=None):
        """Enter in the transcript box: send the line typed there as a question."""
        box = self.trans_box
        line = box.get("insert linestart", "insert lineend")
        text = line.removesuffix("…").strip()
        for label in ("You:", "Them:", "Typed:"):
            text = text.removeprefix(label).strip()
        if text:
            box.delete("insert linestart", "insert lineend +1c")
            self.append(box, f"Typed: {text}\n", "Typed")
            box.see("end")
            self.answerer.jobs.put(("manual", text))
        return "break"   # don't insert a newline

    def answer_last(self):
        """Answer what's typed in the Ask box if anything, otherwise the last question heard."""
        if self.ask.get().strip():
            self.submit_typed()
        else:
            self.answerer.jobs.put(("force", None))

    def edit_last(self):
        """Put the last thing the interviewer said in the Ask box, to fix transcription mistakes."""
        with self.lock:
            last = next((line for line in reversed(self.transcript) if line.startswith("[Them]")), "")
        self.ask.delete(0, "end")
        self.ask.insert(0, last.removeprefix("[Them]").strip())
        self.ask.focus_set()
        self.ask.icursor("end")

    def pick_resume(self):
        path = filedialog.askopenfilename(filetypes=[("Resume", "*.pdf *.txt *.md")])
        if path:
            self.resume = read_resume(path)
            self.status.config(text=f"Resume loaded ({len(self.resume.split())} words)")

    def clear(self):
        with self.lock:
            self.transcript.clear()
            self.answered.clear()
        self.partial_text.clear()
        self.trans_box.delete("1.0", "end")
        self.ans_box.delete("1.0", "end")

    def append(self, box, text, tag=None):
        follow = box.yview()[1] > 0.98
        box.insert("end", text, tag)
        if follow:
            box.see("end")

    def pump(self):
        """Apply events from worker threads to the widgets (Tk is single-threaded)."""
        try:
            while True:
                kind, data = self.ui.get_nowait()
                if kind == "status":
                    self.status.config(text=data, fg="#555")
                elif kind == "partial":
                    label, utt, text = data
                    if utt > self.finalized.get(label, 0):
                        self.partial_text[label] = (utt, text)
                        self.redraw_partials()
                elif kind == "finalized":
                    label, utt = data
                    self.finalized[label] = max(utt, self.finalized.get(label, 0))
                    if self.partial_text.get(label, (0,))[0] <= utt:
                        self.partial_text.pop(label, None)
                        self.redraw_partials()
                elif kind == "transcript":
                    label, text = data
                    live, self.partial_text = self.partial_text, {}
                    self.redraw_partials()
                    self.append(self.trans_box, f"{label}: ", label)
                    self.append(self.trans_box, text + "\n")
                    self.partial_text = live
                    self.redraw_partials()
                elif kind == "answer_start":
                    if self.ans_box.get("1.0", "end").strip():
                        self.append(self.ans_box, "\n" + "─" * 40 + "\n", "sep")
                elif kind == "answer":
                    self.append(self.ans_box, data.replace("**", ""))
                elif kind == "answer_end":
                    self.append(self.ans_box, "\n")
        except queue.Empty:
            pass
        level = max(self.them.level, self.you.level if self.use_mic.get() else 0)
        self.meter.delete("all")
        self.meter.create_rectangle(0, 0, min(80, int(level * 800)), 10, fill="#4a4", width=0)
        self.root.after(50, self.pump)


def hide_from_taskbar(root: tk.Tk) -> None:
    """Windows only: make the window a tool window so it has no taskbar button."""
    import ctypes
    user32 = ctypes.windll.user32
    GWL_EXSTYLE, WS_EX_APPWINDOW, WS_EX_TOOLWINDOW = -20, 0x40000, 0x80
    hwnd = user32.GetParent(root.winfo_id())
    style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE, (style & ~WS_EX_APPWINDOW) | WS_EX_TOOLWINDOW)
    root.withdraw()   # re-show so Windows picks up the new style
    root.after(10, root.deiconify)


def main():
    parser = argparse.ArgumentParser(description="Listen to a call and answer questions live.")
    parser.add_argument("--resume", help="PDF/TXT resume to ground personal answers in")
    args = parser.parse_args()
    root = tk.Tk()
    App(root, args.resume)
    if os.name == "nt":
        root.after(100, hide_from_taskbar, root)
    root.mainloop()


if __name__ == "__main__":
    main()
