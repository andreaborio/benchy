#!/usr/bin/env python3
"""Tests for benchy_common.chat_stream — the streaming client behind the live generation box.
Fully offline: a stdlib SSE server serves canned token deltas on an ephemeral port. The
contract under test is that streaming returns the SAME text a blocking chat() would (so the
scorer is unchanged) while routing reasoning vs answer to on_delta as tokens arrive.

Run:  python3 test_stream.py
"""
import json, os, sys, threading, unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import benchy_common as bc


def sse(chunks, reasoning_field=False, drop_first=False):
    """Build a handler that streams `chunks` as content deltas (or reasoning_content deltas
    if reasoning_field), terminated by [DONE]. drop_first=True truncates ONLY the first
    request (closes after one chunk, no [DONE]) and serves the full stream thereafter, so
    chat_stream's retry path can be observed recovering on the second attempt."""
    state = {"n": 0}
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            state["n"] += 1
            truncate = drop_first and state["n"] == 1
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.end_headers()
            for i, c in enumerate(chunks):
                if truncate and i >= 1:
                    return  # clean early close, no [DONE] -> incomplete stream -> client retries
                key = "reasoning_content" if reasoning_field else "content"
                evt = {"choices": [{"delta": {key: c}}]}
                self.wfile.write(b"data: " + json.dumps(evt).encode() + b"\n\n")
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
    return H


def blocking(content):
    """A non-streaming server (no event-stream content-type): exercises the fallback path."""
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)
    return H


class _Srv:
    def __init__(self, handler):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.base = "http://127.0.0.1:%d" % self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
    def __enter__(self): return self
    def __exit__(self, *a): self.httpd.shutdown(); self.httpd.server_close()


def collect(base, **kw):
    seen = {"reasoning": [], "answer": [], "reset": 0}
    def on_delta(kind, text):
        if kind == "reset": seen["reset"] += 1
        else: seen[kind].append(text)
    full = bc.chat_stream("hi", server_base=base, model="m", on_delta=on_delta, **kw)
    return full, seen


class TestChatStream(unittest.TestCase):
    def test_returns_full_content_like_blocking(self):
        # the returned text is the concatenation of content deltas — identical to chat()
        with _Srv(sse(["Hel", "lo ", "wor", "ld"])) as s:
            full, seen = collect(s.base)
        self.assertEqual(full, "Hello world")
        self.assertEqual("".join(seen["answer"]), "Hello world")
        self.assertEqual(seen["reasoning"], [])

    def test_inline_think_split_across_chunks(self):
        # the <think> tags are split across chunk boundaries — the splitter must still route
        # reasoning vs answer correctly, and the returned content keeps the tags verbatim
        chunks = ["<th", "ink>rea", "son", "ing</thi", "nk>the ans", "wer"]
        with _Srv(sse(chunks)) as s:
            full, seen = collect(s.base, think=True)
        self.assertEqual(full, "<think>reasoning</think>the answer")
        self.assertEqual("".join(seen["reasoning"]), "reasoning")
        self.assertEqual("".join(seen["answer"]), "the answer")

    def test_reasoning_content_field_not_in_returned_text(self):
        # providers that put chain-of-thought in a separate reasoning_content field: it is
        # surfaced to on_delta as reasoning but is NOT part of message.content / the return
        with _Srv(sse(["step 1 ", "step 2"], reasoning_field=True)) as s:
            full, seen = collect(s.base, think=True)
        self.assertEqual(full, "")
        self.assertEqual("".join(seen["reasoning"]), "step 1 step 2")
        self.assertEqual(seen["answer"], [])

    def test_blocking_fallback_when_server_does_not_stream(self):
        # non event-stream response -> one blocking read, still routed through the splitter
        with _Srv(blocking("<think>r</think>a")) as s:
            full, seen = collect(s.base, think=True)
        self.assertEqual(full, "<think>r</think>a")
        self.assertEqual("".join(seen["reasoning"]), "r")
        self.assertEqual("".join(seen["answer"]), "a")

    def test_midstream_drop_retries_and_recovers(self):
        # the first request is truncated (no [DONE]); chat_stream must detect the incomplete
        # stream, signal the box to drop its partial buffer (on_delta('reset','')), retry, and
        # return the FULL text from the clean second attempt — never a truncated answer
        bc_sleep = bc.time.sleep
        bc.time.sleep = lambda *_: None   # don't actually wait the 2s/8s backoff in the test
        try:
            with _Srv(sse(["a", "b", "c"], drop_first=True)) as s:
                full, seen = collect(s.base)
        finally:
            bc.time.sleep = bc_sleep
        self.assertEqual(full, "abc")
        self.assertGreaterEqual(seen["reset"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
