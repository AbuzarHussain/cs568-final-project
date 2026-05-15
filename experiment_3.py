"""
Experiment 3: Prompt Specificity Gradient
Research Question: How does increasing prompt specificity affect output quality and consistency?

Methodology:
  For each of 5 tasks, 4 prompt variants are defined at increasing specificity:
    L1 (vague)       — one sentence, no constraints
    L2 (rough)       — adds approximate length or scope
    L3 (formatted)   — adds explicit structure and item count
    L4 (constrained) — adds audience, a format template, and an example pattern

  Each variant is run N_RUNS times at temperature 0.7.
  All N_RUNS outputs are individually rated by an LLM-as-judge (1-10) on
  relevance, completeness, and structure. Two metrics are derived:
    mean_quality — average judge score across runs (higher = better outputs)
    quality_std  — standard deviation of scores across runs (lower = more consistent quality)

Requirements:
    pip install groq numpy
"""

from dotenv import load_dotenv
from groq import Groq
import numpy as np
import os
import time
import csv

load_dotenv()
API_KEY = os.getenv("API_KEY")
MODEL        = "llama-3.1-8b-instant"
JUDGE_MODEL  = "llama-3.3-70b-versatile"   # stronger model for evaluation
N_RUNS  = 3
TEMP    = 0.7

client = Groq(api_key=API_KEY)

LEVEL_NAMES = ["L1_vague", "L2_rough", "L3_formatted", "L4_constrained"]

TASKS = [
    {
        "task": "Explain Climate Change",
        "prompts": [
            # L1
            "Explain climate change.",
            # L2
            "Explain climate change in a few paragraphs.",
            # L3
            "Explain climate change in exactly 3 bullet points, each exactly one sentence.",
            # L4
            (
                "Explain the 3 main causes of climate change to a high school student. "
                "Use this exact format for each cause — "
                "Cause: [name] | Impact: [one sentence]. "
                "Give exactly 3 causes."
            ),
        ],
    },
    {
        "task": "Recommend Books",
        "prompts": [
            # L1
            "Recommend some books.",
            # L2
            "Recommend a few books I should read.",
            # L3
            "Recommend exactly 3 books. For each, give the title and author.",
            # L4
            (
                "Recommend exactly 3 books for someone who enjoys science fiction. "
                "Format each as: [Number]. '[Title]' by [Author] — [One sentence on why to read it]."
            ),
        ],
    },
    {
        "task": "Productivity Tips",
        "prompts": [
            # L1
            "Give me productivity tips.",
            # L2
            "Give me some tips on how to be more productive at work.",
            # L3
            "Give exactly 4 productivity tips, each as a one-sentence rule.",
            # L4
            (
                "Give exactly 4 productivity tips for a college student studying for exams. "
                "Format each as: Tip [number]: [Short title] — [One sentence explanation of the technique]."
            ),
        ],
    },
    {
        "task": "Explain Machine Learning",
        "prompts": [
            # L1
            "Explain machine learning.",
            # L2
            "Explain what machine learning is in simple terms.",
            # L3
            (
                "Explain machine learning in exactly 2 paragraphs: "
                "first the definition, then a real-world example."
            ),
            # L4
            (
                "Explain machine learning to someone with no technical background. "
                "Use exactly 2 paragraphs. "
                "Paragraph 1: define machine learning using an everyday analogy. "
                "Paragraph 2: give one specific real-world application and explain in one sentence how it works."
            ),
        ],
    },
    {
        "task": "Morning Routine",
        "prompts": [
            # L1
            "Give me a morning routine.",
            # L2
            "Give me a morning routine to start my day well.",
            # L3
            "Give me a morning routine with exactly 5 steps, each one sentence.",
            # L4
            (
                "Give me a productive morning routine for a remote worker. "
                "List exactly 5 steps. "
                "Format each as: Step [number] ([duration]): [One sentence describing the activity and its benefit]."
            ),
        ],
    },
]


def run_prompt(prompt: str, n: int = N_RUNS) -> list[str]:
    outputs = []
    for i in range(n):
        response = client.chat.completions.create(
            model=MODEL,
            temperature=TEMP,
            messages=[{"role": "user", "content": prompt}],
        )
        outputs.append(response.choices[0].message.content.strip())
        print(f"    Run {i + 1}/{n} done.")
        time.sleep(3)
    return outputs


def evaluate_quality(prompt: str, output: str) -> int:
    """LLM-as-judge: chain-of-thought format-compliance scoring, 1-10."""
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

    # Parse "FINAL SCORE: N" from the chain-of-thought response
    for line in reversed(raw.splitlines()):
        if "FINAL SCORE" in line.upper():
            for token in line.split():
                if token.isdigit():
                    return min(max(int(token), 1), 10)
    # Fallback: grab any digit from the last line
    for char in raw[-30:]:
        if char.isdigit():
            return min(max(int(char), 1), 10)
    return 5


def score_all_runs(prompt: str, outputs: list[str]) -> tuple[float, float]:
    """Rate every run and return (mean_quality, quality_std)."""
    scores = []
    for i, output in enumerate(outputs):
        score = evaluate_quality(prompt, output)
        scores.append(score)
        print(f"    Judge scored run {i + 1}: {score}/10")
    return round(float(np.mean(scores)), 4), round(float(np.std(scores)), 4)


def run_experiment() -> list[dict]:
    results = []

    for task_data in TASKS:
        task = task_data["task"]
        print(f"\n{'=' * 60}")
        print(f"TASK: {task}")

        for level_idx, prompt in enumerate(task_data["prompts"]):
            level_num  = level_idx + 1
            level_name = LEVEL_NAMES[level_idx]
            print(f"\n  [L{level_num} — {level_name}]")
            print(f"  Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

            outputs = run_prompt(prompt)

            print("  Scoring all runs with LLM-as-judge...")
            mean_quality, quality_std = score_all_runs(prompt, outputs)

            print(f"  Mean quality: {mean_quality:.2f}/10 | Std dev: {quality_std:.4f}")

            results.append({
                "task":         task,
                "level":        level_num,
                "level_name":   level_name,
                "prompt":       prompt,
                "mean_quality": mean_quality,
                "quality_std":  quality_std,
                "sample_output": outputs[0][:300].replace("\n", " "),
            })

    return results


def print_results(results: list[dict]):
    print("\n\n" + "=" * 70)
    print("RESULTS SUMMARY — per task / level")
    print(f"{'Task':<28} {'Level':<20} {'Mean Quality':>13} {'Std Dev':>8}")
    print("-" * 72)
    for r in results:
        print(
            f"{r['task']:<28} L{r['level']} {r['level_name']:<18} "
            f"{r['mean_quality']:>12.2f} {r['quality_std']:>8.4f}"
        )

    print("\n" + "=" * 55)
    print("AVERAGES BY SPECIFICITY LEVEL")
    print(f"{'Level':<24} {'Avg Mean Quality':>17} {'Avg Std Dev':>12}")
    print("-" * 55)
    for lvl in range(1, 5):
        subset   = [r for r in results if r["level"] == lvl]
        avg_qual = np.mean([r["mean_quality"] for r in subset])
        avg_std  = np.mean([r["quality_std"]  for r in subset])
        print(f"L{lvl} {LEVEL_NAMES[lvl - 1]:<22} {avg_qual:>17.2f} {avg_std:>12.4f}")

    avgs = [
        np.mean([r["mean_quality"] for r in results if r["level"] == lvl])
        for lvl in range(1, 5)
    ]
    trend = "IMPROVES" if avgs[-1] > avgs[0] else "DOES NOT IMPROVE"
    print(
        f"\nConclusion: Output quality {trend} with specificity "
        f"(L1 avg={avgs[0]:.2f} → L4 avg={avgs[3]:.2f})."
    )


def save_results(results: list[dict]):
    with open("specificity_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task", "level", "level_name", "prompt",
                "mean_quality", "quality_std", "sample_output",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    print("Saved results to specificity_results.csv")


if __name__ == "__main__":
    results = run_experiment()
    print_results(results)
    save_results(results)
