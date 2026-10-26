"""Mobile Chat: a phone-friendly chat page, answered by Groq like its web Playground.

Runs a small server on this PC (your API key stays here) and serves a chat page your
phone can open over the same Wi-Fi. Answers render Markdown — headings, tables, code —
and show the model's reasoning in a collapsible section, like the Groq Playground.

    python mobile_chat.py [--port 8600]
"""

import argparse
import json
import os
import secrets
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# No system prompt by default, same as the Playground. Set CHAT_SYSTEM in .env to add one.
SYSTEM = os.getenv("CHAT_SYSTEM", "").strip()
ACCESS_KEY = secrets.token_urlsafe(8)   # only someone with the printed link can use your key

client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")


def stream_reply(messages: list[dict]):
    """Yield ("r", reasoning) and ("t", answer text) chunks as they stream in."""
    extra = {"reasoning_effort": "medium"} if "gpt-oss" in MODEL else {}
    resp = client.chat.completions.create(
        model=MODEL, stream=True, temperature=1, max_completion_tokens=8192,
        messages=([{"role": "system", "content": SYSTEM}] if SYSTEM else []) + messages,
        **extra,
    )
    try:
        for event in resp:
            if not event.choices:
                continue
            delta = event.choices[0].delta
            if getattr(delta, "reasoning", None):
                yield "r", delta.reasoning
            if delta.content:
                yield "t", delta.content
    finally:
        resp.close()


class Handler(BaseHTTPRequestHandler):
    def authorized(self) -> bool:
        key = parse_qs(urlparse(self.path).query).get("key", [""])[0]
        if secrets.compare_digest(key, ACCESS_KEY):
            return True
        self.send_error(403, "Open the link printed in the terminal (it includes ?key=...)")
        return False

    def do_GET(self):
        if urlparse(self.path).path != "/":
            return self.send_error(404)
        if not self.authorized():
            return
        body = PAGE.replace("__MODEL__", MODEL).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if urlparse(self.path).path != "/api/chat":
            return self.send_error(404)
        if not self.authorized():
            return
        length = int(self.headers.get("Content-Length", 0))
        messages = json.loads(self.rfile.read(length))["messages"]
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            for kind, text in stream_reply(messages):
                self.wfile.write((json.dumps({kind: text}) + "\n").encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # phone navigated away mid-answer
        except Exception as e:
            self.wfile.write((json.dumps({"e": str(e)}) + "\n").encode("utf-8"))

    def log_message(self, *_):
        pass


def lan_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("8.8.8.8", 80))   # no packets sent; just picks the Wi-Fi interface
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#f6f5f2">
<title>Groq Chat</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.2/marked.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/dompurify/3.1.6/purify.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
<style>
  :root { --bg:#f6f5f2; --card:#fff; --text:#1d1d1b; --muted:#77756f; --line:#e4e2dc;
          --accent:#f55036; --code:#f3f2ee; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#151514; --card:#1f1f1d; --text:#ecebe6; --muted:#9a988f; --line:#33322f;
            --accent:#ff6a50; --code:#262624; }
    .hljs { background: var(--code) !important; color: var(--text) !important; }
  }
  * { box-sizing: border-box; }
  html, body { margin:0; height:100%; background:var(--bg); color:var(--text);
    font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  body { display:flex; flex-direction:column; height:100dvh; }
  header { display:flex; align-items:center; gap:8px; padding:10px 16px;
    padding-top:calc(10px + env(safe-area-inset-top)); border-bottom:1px solid var(--line); }
  header b { font-size:17px; }
  header small { color:var(--muted); flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  header button { border:1px solid var(--line); background:var(--card); color:var(--text);
    border-radius:8px; padding:6px 12px; font-size:14px; }
  #log { flex:1; overflow-y:auto; padding:16px; -webkit-overflow-scrolling:touch; }
  .msg { max-width:760px; margin:0 auto 18px; }
  .user { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:10px 14px; }
  .label { font-size:12px; letter-spacing:.06em; color:var(--muted); margin-bottom:4px; }
  details { color:var(--muted); font-size:14px; margin:4px 0 8px; }
  details summary { cursor:pointer; }
  details div { white-space:pre-wrap; padding:6px 0 0 14px; border-left:2px solid var(--line); }
  .md { overflow-wrap:anywhere; }
  .md h1 { font-size:1.45em; line-height:1.25; margin:.6em 0 .4em; }
  .md h2 { font-size:1.25em; margin:.9em 0 .4em; }
  .md h3 { font-size:1.08em; margin:.8em 0 .3em; }
  .md p, .md ul, .md ol { margin:.5em 0; }
  .md ul, .md ol { padding-left:1.3em; }
  .md table { border-collapse:collapse; display:block; overflow-x:auto; max-width:100%;
    font-size:14px; margin:.7em 0; }
  .md th, .md td { border:1px solid var(--line); padding:6px 9px; text-align:left; vertical-align:top;
    min-width:110px; }
  .md th { background:var(--code); }
  .md code { background:var(--code); padding:1px 5px; border-radius:4px; font-size:.9em; }
  .md pre { background:var(--code); border-radius:8px; overflow-x:auto; padding:0; }
  .md pre code { display:block; padding:12px; background:none; font-size:13px; }
  .md blockquote { margin:.6em 0; padding-left:12px; border-left:3px solid var(--line); color:var(--muted); }
  .err { color:var(--accent); }
  form { display:flex; gap:8px; padding:10px 12px; padding-bottom:calc(10px + env(safe-area-inset-bottom));
    border-top:1px solid var(--line); background:var(--bg); }
  textarea { flex:1; resize:none; border:1px solid var(--line); border-radius:12px; padding:10px 12px;
    font:inherit; background:var(--card); color:var(--text); max-height:140px; }
  form button { border:0; border-radius:12px; padding:0 16px; background:var(--accent); color:#fff;
    font:600 15px inherit; }
  form button:disabled { opacity:.5; }
  .empty { color:var(--muted); text-align:center; margin-top:30vh; }
</style>
</head>
<body>
<header><b>Chat</b><small>__MODEL__</small><button id="clear" type="button">Clear</button></header>
<div id="log"><div class="empty">Ask anything</div></div>
<form id="f">
  <textarea id="q" rows="1" placeholder="Message" enterkeyhint="send"></textarea>
  <button id="send">Send</button>
</form>
<script>
const KEY = new URLSearchParams(location.search).get("key") || "";
const log = document.getElementById("log"), q = document.getElementById("q"),
      send = document.getElementById("send");
let history = [];
marked.setOptions({ gfm: true, breaks: false });

function md(text) { return DOMPurify.sanitize(marked.parse(text)); }
function scroll() { log.scrollTop = log.scrollHeight; }
function add(html, cls) {
  log.querySelector(".empty")?.remove();
  const d = document.createElement("div"); d.className = "msg " + (cls || ""); d.innerHTML = html;
  log.appendChild(d); scroll(); return d;
}
function esc(s) { const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }

q.addEventListener("input", () => { q.style.height = "auto"; q.style.height = q.scrollHeight + "px"; });
q.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey && window.matchMedia("(pointer:fine)").matches) {
    e.preventDefault(); document.getElementById("f").requestSubmit();
  }
});
document.getElementById("clear").onclick = () => {
  history = []; log.innerHTML = '<div class="empty">Ask anything</div>';
};

document.getElementById("f").addEventListener("submit", async e => {
  e.preventDefault();
  const text = q.value.trim(); if (!text || send.disabled) return;
  q.value = ""; q.style.height = "auto"; send.disabled = true;
  add('<div class="label">USER</div>' + esc(text).replace(/\n/g, "<br>"), "user");
  history.push({ role: "user", content: text });

  const box = add('<div class="label">ASSISTANT</div><details hidden><summary>Reasoning</summary><div></div></details><div class="md">…</div>');
  const details = box.querySelector("details"), rbox = details.querySelector("div"),
        out = box.querySelector(".md");
  let answer = "", reasoning = "", pending = false;
  const render = () => { pending = false; out.innerHTML = md(answer) || "…"; scroll(); };

  try {
    const res = await fetch("/api/chat?key=" + encodeURIComponent(KEY), {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history }),
    });
    if (!res.ok) throw new Error("Server said " + res.status);
    const reader = res.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read(); if (done) break;
      buf += dec.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl); buf = buf.slice(nl + 1);
        if (!line) continue;
        const m = JSON.parse(line);
        if (m.r) { reasoning += m.r; details.hidden = false; rbox.textContent = reasoning; }
        if (m.t) { answer += m.t; if (!pending) { pending = true; requestAnimationFrame(render); } }
        if (m.e) throw new Error(m.e);
      }
    }
    render();
    out.querySelectorAll("pre code").forEach(el => hljs.highlightElement(el));
    history.push({ role: "assistant", content: answer });
  } catch (err) {
    out.innerHTML = '<span class="err">' + esc(err.message) + "</span>";
    history.pop();
  } finally {
    send.disabled = false;
  }
});
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Phone-friendly Groq chat served from this PC.")
    parser.add_argument("--port", type=int, default=8600)
    args = parser.parse_args()
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY not set — add it to .env")
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Model: {MODEL}")
    print(f"On this PC:  http://localhost:{args.port}/?key={ACCESS_KEY}")
    print(f"On phone:    http://{lan_ip()}:{args.port}/?key={ACCESS_KEY}   (same Wi-Fi)")
    print("Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
