#!/usr/bin/env python3
"""Measure RAM<->VRAM context transfer on the box's ninfer server.

Sends N distinct ~150K-token prompts (cold prefill), re-sends one while hot
(device hit), fills the device pool to force an eviction (VRAM->RAM demote),
then re-sends the evicted prompt (RAM->VRAM restore).

Stdlib only. Usage: NINFER_URL=http://host:8000/v1 python3 bench-transfer.py
"""
import json
import os
import time
import urllib.request

BASE = os.environ.get("NINFER_URL", "http://localhost:8000/v1")
MODEL = "qwen3.8-27b"

_SENTENCE = "The quick brown fox jumps over the lazy dog while the winter "
_SENTENCE += "wind whistles through the pines and the snow falls silently. "
TARGET_CHARS = 600_000  # ~150K tokens at ~4 chars/token


def make_prompt(seed):
    text = (_SENTENCE * (TARGET_CHARS // len(_SENTENCE) + 1))[:TARGET_CHARS]
    return "seed=%d run=%d\n" % (seed, int(time.time() // 60)) + text


def send(prompt, label):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8,
    }
    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())
    ttft = None
    for chunk in data.get("choices", []):
        if "prompt_tokens" in chunk:
            ttft = chunk["prompt_tokens"]
    elapsed = time.monotonic() - t0
    usage = data.get("usage", {})
    print(
        f"{label:28s} wall={elapsed:7.2f}s "
        f"prompt={usage.get('prompt_tokens')} cached={usage.get('cached_tokens')} "
        f"comp={usage.get('completion_tokens')}"
    )
    return elapsed, usage


def main():
    prompts = [make_prompt(i) for i in range(5)]
    print(f"pool: 480K device tokens; prompts ~{TARGET_CHARS // 4} tokens each")
    print("--- cold prefill (A) ---")
    send(prompts[0], "A1 cold")
    send(prompts[0], "A2 hot device")
    print("--- fill pool with B, C, D (A+B+C+D ~492K > 480K) ---")
    send(prompts[1], "B cold")
    send(prompts[2], "C cold")
    send(prompts[3], "D cold")
    print("--- E cold forces full eviction of LRU (A -> host) ---")
    send(prompts[4], "E cold + demote A")
    print("--- A again: now on host, measures RAM->VRAM restore ---")
    send(prompts[0], "A3 restore from RAM")
    print("--- A once more: should be hot again on device ---")
    send(prompts[0], "A4 hot device")


if __name__ == "__main__":
    main()
