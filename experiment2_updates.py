"""
Experiment 2: Prompt Misconception Detector (Updated Evaluation)
Extends: "Why Johnny Can't Prompt"

Detects common misconceptions in user prompts using rule-based pattern
matching and then uses Groq to generate improved prompts.

Evaluation (updated):
  - Runs both the original and improved prompt through LLaMA 3.1 8B
  - Scores both outputs using LLaMA 3.3 70B (stronger judge, mirrors Exp 3)
  - Uses chain-of-thought rubric for discrimination (mirrors Exp 3's judge)
  - Saves before/after scores to exp2_evaluation_results.csv
  - Generates a bar chart comparing original vs improved prompt output quality

Requirements: pip install groq numpy matplotlib python-dotenv
"""

import re
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY     = os.getenv("API_KEY")
MODEL       = "llama-3.1-8b-instant"       # model being tested
JUDGE_MODEL = "llama-3.3-70b-versatile"    # stronger judge — mirrors Exp 3

client = Groq(api_key=API_KEY)

RULES = [
    {
        "name": "Too vague / too short",
        "pattern": r"^.{0,40}$",
        "explanation": "The prompt is very short and likely too vague for the model to give a useful, consistent response.",
    },
    {
        "name": "Asking to browse the web",
        "pattern": r"\b(search the web|browse|google|look it up|find online|search online|go to website)\b",
        "explanation": "LLMs cannot browse the internet or access live websites. They only know what was in their training data.",
    },
    {
        "name": "Asking to remember previous conversations",
        "pattern": r"\b(remember (when|last time|before|earlier|previously|what I said)|as I (said|mentioned|told you)|from (our|the) last (chat|conversation|session))\b",
        "explanation": "LLMs have no memory between separate conversations. Each session starts fresh with no knowledge of past interactions.",
    },
    {
        "name": "Asking for real-time information",
        "pattern": r"\b(current (price|weather|news|score|stock)|right now|today'?s? (weather|news|price)|live (score|update|feed)|latest news|what is happening now)\b",
        "explanation": "LLMs do not have access to real-time data. They cannot tell you today's weather, live stock prices, or breaking news.",
    },
    {
        "name": "Asking for specific URLs or links",
        "pattern": r"\b(give me (a |the )?link|send me (a |the )?url|provide (a |the )?link|what is the url|share (a |the )?link)\b",
        "explanation": "LLMs cannot generate or verify real URLs. Any links they produce may be hallucinated or broken.",
    },
    {
        "name": "Asking to execute or run code",
        "pattern": r"\b(run this code|execute (this|the) (code|script|program)|compile (this|the)|test this (code|script))\b",
        "explanation": "LLMs cannot execute code. They can only read, write, and explain it. Use a code interpreter or IDE to run code.",
    },
    {
        "name": "No clear task specified",
        "pattern": r"^(tell me|what about|how about|thoughts on|opinion on|talk about|discuss) .{0,30}$",
        "explanation": "The prompt lacks a clear, specific task. Vague prompts produce unpredictable and inconsistent outputs.",
    },
    {
        "name": "Asking for guaranteed exact word/character count",
        "pattern": r"\b(exactly \d+ words|precisely \d+ words|must be \d+ words|no more than \d+ words|under \d+ words)\b",
        "explanation": "LLMs cannot reliably hit exact word counts. They can approximate length but will rarely match a precise number.",
    },
    {
        "name": "Asking for personal opinions as facts",
        "pattern": r"\b(what do you (really |truly |honestly )?(think|feel|believe|prefer)|your (personal )?opinion|do you (like|prefer|enjoy))\b",
        "explanation": "LLMs do not have personal opinions or feelings. They can present multiple perspectives but not genuine personal views.",
    },
    {
        "name": "Asking to access files or external systems",
        "pattern": r"\b(access my (files?|computer|drive|database|email|calendar)|read my (files?|documents?|emails?)|open (my |the )?(file|folder|drive))\b",
        "explanation": "LLMs cannot access your local files, computer, or external systems unless explicitly given tools to do so.",
    },
]


# ── Detection & suggestion ───────────────────────────────────────────────────

def detect_misconceptions(prompt: str) -> list[dict]:
    flags = []
    for rule in RULES:
        if re.search(rule["pattern"], prompt, re.IGNORECASE):
            flags.append(rule)
    return flags


def get_groq_suggestion(prompt: str, flags: list[dict]) -> str:
    """Generate an improved prompt using LLaMA 3.1 8B."""
    if flags:
        issues = "\n".join(f"- {f['name']}: {f['explanation']}" for f in flags)
        system_message = (
            "You are an expert at helping non-technical users write better prompts for AI models. "
            "You will be given a user's prompt and a list of detected issues with it. "
            "Your job is to:\n"
            "1. Briefly explain why each issue is a problem in simple terms\n"
            "2. Rewrite the prompt to fix all the issues\n"
            "Keep your response concise and friendly. No bullet point walls — be direct.\n\n"
            "IMPORTANT: End your response with a clearly labeled section:\n"
            "IMPROVED PROMPT: <the rewritten prompt only, no extra text>"
        )
        user_message = (
            f"User's prompt: \"{prompt}\"\n\n"
            f"Detected issues:\n{issues}\n\n"
            f"Please explain the issues simply and provide a rewritten, improved prompt."
        )
    else:
        system_message = (
            "You are an expert at helping non-technical users write better prompts for AI models. "
            "No major issues were detected in the prompt, but you should still suggest improvements. "
            "Rate the prompt quality out of 10 and suggest one or two specific ways to make it even better.\n\n"
            "IMPORTANT: End your response with a clearly labeled section:\n"
            "IMPROVED PROMPT: <the rewritten prompt only, no extra text>"
        )
        user_message = (
            f"User's prompt: \"{prompt}\"\n\n"
            f"No major issues detected. Please rate it and suggest improvements."
        )

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.5,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user",   "content": user_message},
        ],
    )
    return response.choices[0].message.content.strip()


def extract_improved_prompt(suggestion: str, original: str) -> str:
    """
    Pull the rewritten prompt out of the suggestion text.
    Falls back to the full suggestion if the label isn't found.
    """
    for line in suggestion.splitlines():
        if line.strip().upper().startswith("IMPROVED PROMPT:"):
            extracted = line.split(":", 1)[1].strip().strip('"').strip("'")
            if extracted:
                return extracted
    # Fallback: last non-empty line
    lines = [l.strip() for l in suggestion.splitlines() if l.strip()]
    return lines[-1] if lines else original


# ── Output generation ────────────────────────────────────────────────────────

def run_prompt_get_output(prompt: str) -> str:
    """Run a prompt once and return the output."""
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    time.sleep(3)
    return response.choices[0].message.content.strip()


# ── Stronger judge (mirrors Exp 3) ───────────────────────────────────────────

def evaluate_output_quality(prompt: str, output: str) -> int:
    """
    Score an output 1-10 using LLaMA 3.3 70B with the same chain-of-thought
    rubric used in Experiment 3.
    """
    judge_prompt = (
        f"You are a strict research evaluator scoring AI responses on a 1-10 scale. "
        f"Your scores must reflect real differences — do not cluster around one number.\n\n"
        f"Prompt given to the AI:\n\"{prompt}\"\n\n"
        f"AI response:\n{output}\n\n"
        f"Step 1 — List every explicit constraint in the prompt (exact item counts, "
        f"required format templates, specified audience, length requirements, etc.). "
        f"If there are none, write 'No constraints'.\n\n"
        f"Step 2 — For each constraint, state whether the response MET or VIOLATED it "
        f"and why.\n\n"
        f"Step 3 — Assign a final score using this scale:\n"
        f"  1-2: Off-topic or completely ignores the prompt\n"
        f"  3-4: On-topic but ignores most format/structure constraints\n"
        f"  5-6: Decent content, misses several constraints (wrong count, no template, etc.)\n"
        f"  7-8: Meets most constraints with only minor deviations\n"
        f"  9-10: Meets every constraint exactly; highly specific and complete\n\n"
        f"Important: a vague prompt that got a vague (but valid) response is a 5, not a 9. "
        f"Only give 9-10 when every stated constraint is satisfied.\n\n"
        f"End your response with exactly this line:\nFINAL SCORE: <integer>"
    )
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0.0,
        messages=[{"role": "user", "content": judge_prompt}],
    )
    raw = response.choices[0].message.content.strip()
    time.sleep(2)

    for line in reversed(raw.splitlines()):
        if "FINAL SCORE" in line.upper():
            for token in line.split():
                if token.isdigit():
                    return min(max(int(token), 1), 10)
    for char in raw[-30:]:
        if char.isdigit():
            return min(max(int(char), 1), 10)
    return 5


# ── Save & plot ──────────────────────────────────────────────────────────────

def save_results(rows: list[dict]):
    with open("exp2_evaluation_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt", "detected_issues",
                "improved_prompt", "suggestion",
                "original_score", "improved_score", "improvement",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print("Saved results to exp2_evaluation_results.csv")


def plot_results(rows: list[dict]):
    labels          = [r["prompt"][:35] + "…" if len(r["prompt"]) > 35 else r["prompt"] for r in rows]
    original_scores = [r["original_score"]  for r in rows]
    improved_scores = [r["improved_score"]  for r in rows]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))
    bars1 = ax.bar(x - width / 2, original_scores, width, label="Original prompt output",  color="#7bafd4")
    bars2 = ax.bar(x + width / 2, improved_scores, width, label="Improved prompt output", color="#2c5f8a")

    # Avg lines
    avg_orig = np.mean(original_scores)
    avg_impr = np.mean(improved_scores)
    ax.axhline(avg_orig, color="#7bafd4", linestyle="--", linewidth=1.2, label=f"Avg original ({avg_orig:.2f})")
    ax.axhline(avg_impr, color="#2c5f8a", linestyle="--", linewidth=1.2, label=f"Avg improved ({avg_impr:.2f})")

    ax.set_xlabel("Prompt")
    ax.set_ylabel("Output Quality Score (LLM-as-judge, 1–10)")
    ax.set_title("Experiment 2: Original vs. Improved Prompt Output Quality\n(Judge: LLaMA 3.3 70B)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 10.5)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig("exp2_before_after.png", dpi=150)
    print("Saved chart to exp2_before_after.png")
    plt.show()


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_prompts = [
        "Search the web for the latest iPhone price and send me the link.",
        "Remember last time I asked you about my project? Continue from there.",
        "What's the weather like right now in Chicago?",
        "Run this Python script for me and tell me the output.",
        "Tell me about dogs.",
        "Explain the causes of the French Revolution.",
        "Give me tips for improving my writing skills.",
        "Summarize the main ideas of machine learning in simple terms.",
    ]

    rows = []

    for prompt in test_prompts:
        print(f"\n{'=' * 60}")
        print(f"PROMPT: \"{prompt}\"")

        # 1. Detect misconceptions
        flags = detect_misconceptions(prompt)
        if flags:
            print(f"  {len(flags)} misconception(s) detected: {', '.join(f['name'] for f in flags)}")
        else:
            print("  No major misconceptions detected.")

        # 2. Generate suggestion + extract improved prompt
        suggestion     = get_groq_suggestion(prompt, flags)
        improved_prompt = extract_improved_prompt(suggestion, prompt)
        print(f"  Improved prompt: \"{improved_prompt[:80]}{'…' if len(improved_prompt) > 80 else ''}\"")

        # 3. Run both original and improved prompt to get outputs
        print("  Running original prompt...")
        original_output = run_prompt_get_output(prompt)

        print("  Running improved prompt...")
        improved_output = run_prompt_get_output(improved_prompt)

        # 4. Score both outputs with the stronger judge (mirrors Exp 3)
        print("  Scoring original output with judge...")
        original_score = evaluate_output_quality(prompt, original_output)
        print(f"    Original score: {original_score}/10")

        print("  Scoring improved output with judge...")
        improved_score = evaluate_output_quality(improved_prompt, improved_output)
        print(f"    Improved score: {improved_score}/10")

        rows.append({
            "prompt":          prompt,
            "detected_issues": "; ".join(f["name"] for f in flags) if flags else "None",
            "improved_prompt": improved_prompt,
            "suggestion":      suggestion,
            "original_score":  original_score,
            "improved_score":  improved_score,
            "improvement":     round(improved_score - original_score, 4),
        })

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print(f"{'Prompt':<45} {'Original':>9} {'Improved':>9} {'Delta':>7}")
    print("-" * 72)
    for r in rows:
        label = r["prompt"][:43] + "…" if len(r["prompt"]) > 43 else r["prompt"]
        print(f"{label:<45} {r['original_score']:>9} {r['improved_score']:>9} {r['improvement']:>+7}")

    avg_orig = np.mean([r["original_score"] for r in rows])
    avg_impr = np.mean([r["improved_score"] for r in rows])
    print(f"\nAverage original score : {avg_orig:.2f}/10")
    print(f"Average improved score : {avg_impr:.2f}/10")
    print(f"Average improvement    : {avg_impr - avg_orig:+.2f}")

    save_results(rows)
    plot_results(rows)