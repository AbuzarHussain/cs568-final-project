"""
Experiment 2: Prompt Misconception Detector
Extends: "Why Johnny Can't Prompt"

Detects common misconceptions in user prompts using rule-based pattern
matching and then uses Groq to generate improved prompts.

After the main run, an optional LLM-as-judge step (mirroring Experiment 3)
rates each improvement suggestion on a 1-10 scale and saves the scores to
exp2_evaluation_results.csv.

Requirements: pip install groq numpy python-dotenv
"""

import re
import csv
import time
import numpy as np
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("API_KEY")
MODEL   = "llama-3.1-8b-instant"

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


def detect_misconceptions(prompt: str) -> list[dict]:
    flags = []
    for rule in RULES:
        if re.search(rule["pattern"], prompt, re.IGNORECASE):
            flags.append(rule)
    return flags


def get_groq_suggestion(prompt: str, flags: list[dict]) -> str:
    if flags:
        issues = "\n".join(
            f"- {f['name']}: {f['explanation']}" for f in flags
        )
        system_message = (
            "You are an expert at helping non-technical users write better prompts for AI models. "
            "You will be given a user's prompt and a list of detected issues with it. "
            "Your job is to: \n"
            "1. Briefly explain why each issue is a problem in simple terms\n"
            "2. Rewrite the prompt to fix all the issues\n"
            "Keep your response concise and friendly. No bullet point walls — be direct."
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
            "Rate the prompt quality out of 10 and suggest one or two specific ways to make it even better. "
            "Keep your response concise and friendly."
        )
        user_message = f"User's prompt: \"{prompt}\"\n\nNo major issues detected. Please rate it and suggest improvements."

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.5,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user",   "content": user_message},
        ]
    )
    return response.choices[0].message.content.strip()


def analyze_prompt(prompt: str):
    print(f"PROMPT: \"{prompt}\"")

    flags = detect_misconceptions(prompt)

    if flags:
        print(f"\n {len(flags)}  MISCONCEPTION(S) DETECTED:")
        for f in flags:
            print(f" {f['name']}")
    else:
        print("\n No major misconceptions detected.")

    print("\n SUGGESTION:")
    suggestion = get_groq_suggestion(prompt, flags)
    print(suggestion)
    print()

def save_results(rows):
    with open("misconception_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["prompt", "detected_issues", "suggestion"]
        )
        writer.writeheader()
        writer.writerows(rows)


# ── Exp3-style LLM-as-judge evaluation ──────────────────────────────────────

def evaluate_suggestion(prompt: str, detected_issues: str, suggestion: str) -> int:
    """Rate an exp2 suggestion 1-10 using an LLM-as-judge (mirrors exp3's judge)."""
    judge_prompt = (
        "You are evaluating the quality of an AI prompt improvement suggestion.\n\n"
        f"Original prompt: \"{prompt}\"\n\n"
        f"Detected issues: {detected_issues}\n\n"
        f"Improvement suggestion:\n{suggestion}\n\n"
        "Rate this suggestion from 1 to 10 based on:\n"
        "- Correctness: does it accurately identify and explain the issues?\n"
        "- Actionability: does the rewritten prompt actually fix the problems?\n"
        "- Clarity: is the advice easy for a non-technical user to understand?\n\n"
        "Reply with ONLY a single integer from 1 to 10. No explanation."
    )
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.0,
        messages=[{"role": "user", "content": judge_prompt}],
    )
    raw = response.choices[0].message.content.strip()
    time.sleep(2)
    for char in raw:
        if char.isdigit():
            return min(max(int(char), 1), 10)
    return 5


def run_evaluation(rows: list[dict]):
    """Score every exp2 suggestion with the LLM-as-judge and save results."""
    print("\n" + "=" * 60)
    print("EXP3-STYLE EVALUATION OF EXP2 SUGGESTIONS")
    print("=" * 60)

    eval_rows = []
    scores = []

    for row in rows:
        score = evaluate_suggestion(
            row["prompt"], row["detected_issues"], row["suggestion"]
        )
        scores.append(score)
        eval_rows.append({
            "prompt":          row["prompt"],
            "detected_issues": row["detected_issues"],
            "suggestion":      row["suggestion"],
            "judge_score":     score,
        })
        print(f"  [{score:>2}/10] {row['prompt'][:70]}")

    avg = round(float(np.mean(scores)), 4)
    print(f"\nAverage judge score across all suggestions: {avg:.2f}/10")

    with open("exp2_evaluation_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["prompt", "detected_issues", "suggestion", "judge_score"],
        )
        writer.writeheader()
        writer.writerows(eval_rows)
    print("Saved evaluation results to exp2_evaluation_results.csv")


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
        flags = detect_misconceptions(prompt)
        suggestion = get_groq_suggestion(prompt, flags)

        analyze_prompt(prompt)

        rows.append({
            "prompt": prompt,
            "detected_issues": "; ".join([f["name"] for f in flags]) if flags else "None",
            "suggestion": suggestion,
        })

    save_results(rows)
    print("Saved results to misconception_results.csv")

    run_evaluation(rows)