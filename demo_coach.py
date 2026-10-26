"""Demo Coach: a presenter-only helper for live client product demos.

It listens to the call like live_assistant.py, and when a client asks a question it
streams a suggested answer grounded in your product docs. The coach window is excluded
from screen capture (Windows 10 2004+), so clients watching your screen share see
only your product while you still see the coach — like a presenter view.

    python demo_coach.py [--docs product.pdf faq.md ...]
"""

import argparse
import ctypes
import os
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, scrolledtext

from live_assistant import API_KEY_VAR, Answerer, App, PROVIDER_NAME

SAVED_DOCS = Path(__file__).parent / "product_docs.txt"   # last loaded docs, reused on next start
MAX_DOC_CHARS = 200_000                                    # keep prompts a sane size

WDA_NONE, WDA_EXCLUDEFROMCAPTURE = 0x0, 0x11

SYSTEM = """You are an expert assistant helping the presenter answer questions live on a client
call. You get a rolling transcript: lines tagged [Them] are the client's attendees, [You] is the
presenter. Transcripts come from speech recognition, so expect missing punctuation and misheard
words (e.g. "Jango" = Django) — infer the intended meaning.

Answer like a great ChatGPT answer: practical and concrete, never vague or textbook-generic.
Use your full general and technical knowledge, and the background material (<product_docs>)
where relevant. The presenter will read your words aloud, so the spoken part must sound like a
person talking: first person ("I", "we"), natural spoken English, no jargon dumps.

Answer format (plain text — the window can't render Markdown, so no ** or # symbols):

Q: <the question, cleaned up>

INTERVIEW ANSWER:
"<3-5 sentences in quotes, exactly what the presenter should say out loud: what it is, why
it's used, and how it works — clear and confident, the way a strong candidate would explain it.>"

SIMPLE EXAMPLE:
<A concrete, everyday scenario that makes it click. Contrast "without it" vs "with it" when that
fits, e.g. "Suppose you have a Java application. Without Docker you'd install Java, Maven and
MySQL on every server... With Docker you build one image and run it anywhere." Add a short
command or code snippet (indented 4 spaces) when it helps, e.g. `docker build` / `docker run`.>

KEY POINTS TO MENTION:
- <2-4 short points that show depth: trade-offs, common follow-ups, real-world gotchas>

FROM MY EXPERIENCE:
<One or two sentences tying it to a real project in the background material, phrased as the
presenter would say it. Include this section ONLY if the material explicitly names this exact
technology or concept (e.g. it lists Redis, so "I used Redis at..." is fine). If the material
doesn't mention it, OMIT the whole section — never claim the presenter used something the
material doesn't list, and never invent numbers, results, or projects.
Only restate what the material actually says. If it lists the skill but not where or how it was
used, keep it general ("Redis is part of my backend stack, so I'm comfortable using it for
caching") — don't make up which project used it or what it achieved. Never include a
percentage or metric unless that exact number appears in the material.>

Special cases:
- Several topics in one question (e.g. "explain Docker, Kubernetes, Terraform"): give
  "1. DOCKER", "2. KUBERNETES", ... each with its own INTERVIEW ANSWER and SIMPLE EXAMPLE, then
  finish with "HOW THEY WORK TOGETHER:" in 2-3 spoken sentences.
- Behavioral / "tell me about a time": the INTERVIEW ANSWER is a spoken STAR story (Situation,
  Task, Action, Result) built on a real experience from the background material; skip SIMPLE
  EXAMPLE.
- Coding / DSA: INTERVIEW ANSWER explains the approach and complexity out loud; SIMPLE EXAMPLE
  walks through a small input step by step, plus short code if useful.
- If showing something on screen would help, end with "SHOW: <what to demo>".
- Keep a single-topic answer around 200-300 words.

Don't invent facts about the presenter or company that the material doesn't support —
employers, projects, customer names, pricing, certifications, or metrics. General knowledge,
best practices, and illustrative examples are fine; just don't present them as the presenter's
own history."""


def set_capture_hidden(root: tk.Tk, hidden: bool) -> bool:
    """Exclude the window from screen capture / screen share (it stays visible locally)."""
    if sys.platform != "win32":
        return False
    hwnd = ctypes.windll.user32.GetParent(root.winfo_id())  # Tk's toplevel frame
    return bool(ctypes.windll.user32.SetWindowDisplayAffinity(
        hwnd, WDA_EXCLUDEFROMCAPTURE if hidden else WDA_NONE))


def read_docs(paths: list[str]) -> str:
    parts = []
    for path in paths:
        if path.lower().endswith(".pdf"):
            from pypdf import PdfReader
            text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        else:
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
        parts.append(f"## {Path(path).name}\n{text.strip()}")
    docs = "\n\n".join(parts)[:MAX_DOC_CHARS]
    SAVED_DOCS.write_text(docs, encoding="utf-8")
    return docs


class DemoAnswerer(Answerer):
    system = SYSTEM
    mode = "demo"

    def context_blocks(self) -> str:
        app = self.app
        client = app.context.get().strip()
        return ((f"<product_docs>\n{app.docs}\n</product_docs>\n\n" if app.docs else "")
                + (f"<client_context>\n{client}\n</client_context>\n\n" if client else ""))


class DemoApp(App):
    """Same listening/answering loop as App, with docs instead of a resume and a private window."""

    def __init__(self, root: tk.Tk, doc_paths: list[str] | None, visible: bool = False):
        self.root = root
        self.ui: queue.Queue = queue.Queue()
        self.audio: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.transcript: list[str] = []
        self.answered: list[str] = []
        if doc_paths:
            self.docs = read_docs(doc_paths)
        elif SAVED_DOCS.exists():
            self.docs = SAVED_DOCS.read_text(encoding="utf-8")
        else:
            self.docs = ""

        root.title("Demo Coach")
        root.geometry("520x720")
        root.attributes("-topmost", True)

        bar = tk.Frame(root)
        bar.pack(fill="x", padx=8, pady=(8, 4))
        self.auto = tk.BooleanVar(value=True)
        self.use_mic = tk.BooleanVar(value=True)
        self.on_top = tk.BooleanVar(value=True)
        self.private = tk.BooleanVar(value=not visible)
        tk.Checkbutton(bar, text="Auto-answer", variable=self.auto).pack(side="left")
        tk.Checkbutton(bar, text="Hear my mic", variable=self.use_mic,
                       command=self.toggle_mic).pack(side="left")
        tk.Checkbutton(bar, text="On top", variable=self.on_top,
                       command=lambda: root.attributes("-topmost", self.on_top.get())).pack(side="left")
        tk.Checkbutton(bar, text="Hide from screen share", variable=self.private,
                       command=self.apply_privacy).pack(side="left")

        tools = tk.Frame(root)
        tools.pack(fill="x", padx=8)
        tk.Button(tools, text="Answer last", command=lambda: self.answerer.jobs.put(("force", None))
                  ).pack(side="left")
        tk.Button(tools, text="Product docs…", command=self.pick_docs).pack(side="left", padx=4)
        tk.Button(tools, text="Clear", command=self.clear).pack(side="left")
        self.privacy_label = tk.Label(tools, font=("Segoe UI", 9, "bold"))
        self.privacy_label.pack(side="right")

        ctx = tk.Frame(root)
        ctx.pack(fill="x", padx=8, pady=(4, 0))
        tk.Label(ctx, text="Client:").pack(side="left")
        self.context = tk.StringVar()
        tk.Entry(ctx, textvariable=self.context).pack(side="left", fill="x", expand=True)

        status = tk.Frame(root)
        status.pack(fill="x", padx=8, pady=4)
        self.status = tk.Label(status, text="Starting...", anchor="w", fg="#555")
        self.status.pack(side="left", fill="x", expand=True)
        self.meter = tk.Canvas(status, width=80, height=10, bg="#ddd", highlightthickness=0)
        self.meter.pack(side="right")

        panes = tk.PanedWindow(root, orient="vertical", sashwidth=6)
        panes.pack(fill="both", expand=True, padx=8)
        self.trans_box = scrolledtext.ScrolledText(panes, height=6, wrap="word", font=("Segoe UI", 9))
        self.ans_box = scrolledtext.ScrolledText(panes, wrap="word", font=("Segoe UI", 12))
        self.trans_box.tag_config("You", foreground="#2a6")
        self.trans_box.tag_config("Them", foreground="#36c")
        self.ans_box.tag_config("sep", foreground="#aaa")
        panes.add(self.trans_box, minsize=60)
        panes.add(self.ans_box, minsize=200)

        ask = tk.Frame(root)
        ask.pack(fill="x", padx=8, pady=8)
        tk.Label(ask, text="Ask:").pack(side="left")
        self.ask = tk.Entry(ask, font=("Segoe UI", 11))
        self.ask.pack(side="left", fill="x", expand=True)
        self.ask.bind("<Return>", self.submit_typed)

        if not os.getenv(API_KEY_VAR):
            self.status.config(text=f"{API_KEY_VAR} not set — add it to .env", fg="red")

        self.answerer = DemoAnswerer(self.ui, self)
        self.answerer.start()
        self.start_audio()
        self.toggle_mic()  # the mic takes a few seconds to open, so start it now rather than on click
        if self.docs:
            self.ui.put(("status", f"Product docs loaded ({len(self.docs.split())} words) — {PROVIDER_NAME}"))
        else:
            self.ui.put(("status", "No product docs loaded — click Product docs… so answers are accurate"))
        root.after(200, self.apply_privacy)  # the window needs to exist before its affinity is set
        self.pump()

    def apply_privacy(self):
        want = self.private.get()
        ok = set_capture_hidden(self.root, want)
        if want and ok:
            self.privacy_label.config(text="● Hidden from share", fg="#2a6")
        elif want:
            self.privacy_label.config(text="● VISIBLE in share (needs Win10 2004+)", fg="red")
        else:
            self.privacy_label.config(text="● Visible in share", fg="#c80")

    def pick_docs(self):
        paths = filedialog.askopenfilenames(filetypes=[("Product docs", "*.pdf *.txt *.md")])
        if paths:
            self.docs = read_docs(list(paths))
            self.status.config(text=f"Product docs loaded ({len(self.docs.split())} words)")


def main():
    parser = argparse.ArgumentParser(description="Presenter-only coach for live product demos.")
    parser.add_argument("--docs", nargs="+", help="PDF/TXT/MD product docs to ground answers in")
    parser.add_argument("--visible", action="store_true", help="start visible in screen share")
    args = parser.parse_args()
    root = tk.Tk()
    DemoApp(root, args.docs, args.visible)
    root.mainloop()


if __name__ == "__main__":
    main()
