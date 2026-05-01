"""
Experiment 1: Consistency Test
Research Question: Do prompts with explicit format instructions produce more consistent LLM outputs?

Requirements:
    pip install groq scikit-learn numpy
"""

from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import time

API_KEY = "groq_api_key"
MODEL   = "llama-3.1-8b-instant"
N_RUNS  = 3                          
TEMP    = 0.7                        

client = Groq(api_key=API_KEY)

# PROMPT_PAIRS = [
#     {
#         "task": "Summarization",
#         "without": "Summarize this text: The Amazon rainforest is one of the world's most biodiverse regions, home to millions of species of plants, animals, and insects. It plays a critical role in regulating the Earth's climate by absorbing vast amounts of carbon dioxide.",
#         "with":    "Summarize this text in exactly 3 bullet points: The Amazon rainforest is one of the world's most biodiverse regions, home to millions of species of plants, animals, and insects. It plays a critical role in regulating the Earth's climate by absorbing vast amounts of carbon dioxide.",
#     },
#     {
#         "task": "Explanation",
#         "without": "Explain what machine learning is.",
#         "with":    "Explain what machine learning is in 2 short paragraphs. First paragraph: definition. Second paragraph: a real-world example.",
#     },
#     {
#         "task": "Advice",
#         "without": "Give me tips for studying better.",
#         "with":    "Give me exactly 4 tips for studying better. Format each tip as: Tip [number]: [Title] - [One sentence explanation].",
#     },
# ]


PROMPT_PAIRS = [
    {
        "task": "Recipe Steps",
        "without": "How do I make scrambled eggs?",
        "with":    "How do I make scrambled eggs? Give exactly 5 numbered steps. Each step should be one sentence.",
    },
    {
        "task": "Study Plan",
        "without": "How should I study for an exam?",
        "with":    "How should I study for an exam? Give exactly 5 numbered tips. Each tip should be one sentence.",
    },
    {
        "task": "Exercise Routine",
        "without": "Give me a morning workout routine.",
        "with":    "Give me a morning workout routine. List exactly 5 exercises. Format each as: [Exercise Name] - [Duration] - [What it targets].",
    },
    {
        "task": "Coding Steps",
        "without": "How do I set up a Python virtual environment?",
        "with":    "How do I set up a Python virtual environment? Give exactly 4 numbered steps. Each step should include the exact command to run.",
    },
    {
        "task": "Travel Packing",
        "without": "What should I pack for a beach trip?",
        "with":    "What should I pack for a beach trip? Give exactly 3 categories with 3 items each. Format: Category: item1, item2, item3.",
    },

    {
        "task": "Compare Python vs JavaScript",
        "without": "Compare Python and JavaScript.",
        "with":    "Compare Python and JavaScript. Give exactly 4 differences. Format each as: [Aspect]: Python does X, JavaScript does Y.",
    },
    {
        "task": "Compare Remote vs Office Work",
        "without": "What are the pros and cons of working from home?",
        "with":    "What are the pros and cons of working from home? List exactly 3 pros and 3 cons. Format: Pro 1: ... Con 1: ...",
    },
    {
        "task": "Compare iOS vs Android",
        "without": "What are the differences between iOS and Android?",
        "with":    "What are the differences between iOS and Android? Give exactly 4 differences. Format each as: [Category] - iOS: [X], Android: [Y].",
    },

    # ── SUMMARIZATION TASKS ─────────────────────────────────────────────
    {
        "task": "Summarize Climate Change",
        "without": "Summarize the main causes of climate change.",
        "with":    "Summarize the main causes of climate change in exactly 3 bullet points. Each bullet point must be exactly one sentence.",
    },
    {
        "task": "Summarize AI risks",
        "without": "What are the risks of artificial intelligence?",
        "with":    "What are the risks of artificial intelligence? List exactly 4 risks. Format each as: Risk [number]: [Name] - [One sentence explanation].",
    },
    {
        "task": "Summarize a Historical Event",
        "without": "Summarize the causes of World War 1.",
        "with":    "Summarize the causes of World War 1 in exactly 3 bullet points. Each bullet point must be one sentence starting with a cause name.",
    },
    {
        "task": "Explain Blockchain",
        "without": "Explain what blockchain is.",
        "with":    "Explain what blockchain is in exactly 2 paragraphs. Paragraph 1: the definition. Paragraph 2: a real-world example.",
    },
    {
        "task": "Explain Supply and Demand",
        "without": "Explain supply and demand.",
        "with":    "Explain supply and demand in exactly 2 paragraphs. Paragraph 1: what supply and demand means. Paragraph 2: a simple everyday example.",
    },
    {
        "task": "Book Recommendations",
        "without": "Recommend some books I should read.",
        "with":    "Recommend exactly 3 books. Format each as: [Title] by [Author] - [One sentence on why to read it].",
    },
    {
        "task": "Productivity Tips",
        "without": "How can I be more productive?",
        "with":    "Give exactly 4 productivity tips. Format each as: Tip [number]: [Title] - [One sentence explanation].",
    },
]


# def run_prompt(prompt: str, n: int = N_RUNS) -> list[str]:
#     outputs = []
#     for i in range(n):
#         response = client.chat.completions.create(
#             model=MODEL,
#             temperature=TEMP,
#             messages=[{"role": "user", "content": prompt}]
#         )
#         output = response.choices[0].message.content.strip()
#         outputs.append(output)
#         print(f"  Run {i+1}/{n} done.")
#     return outputs

def run_prompt(prompt: str, n: int = N_RUNS) -> list[str]:
    outputs = []
    for i in range(n):
        response = client.chat.completions.create(
            model=MODEL,
            temperature=TEMP,
            messages=[{"role": "user", "content": prompt}]
        )
        output = response.choices[0].message.content.strip()
        outputs.append(output)
        print(f"  Run {i+1}/{n} done.")
        time.sleep(5)  # add this line — waits 5 seconds between each call
    return outputs


def measure_consistency(outputs: list[str]) -> float:
    """
    1.0 = perfectly consistent, 0.0 = completely different.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(outputs)
    sim_matrix = cosine_similarity(tfidf_matrix)

    # Average all pairwise similarities
    n = len(outputs)
    total, count = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += sim_matrix[i][j]
            count += 1

    return round(total / count, 4) if count > 0 else 0.0


def run_experiment():
    results = []

    for pair in PROMPT_PAIRS:
        print(f"\n{'='*50}")
        print(f"Task: {pair['task']}")

        print(f"\nRunning WITHOUT format instructions...")
        outputs_without = run_prompt(pair["without"])
        score_without   = measure_consistency(outputs_without)

        print(f"\nRunning WITH format instructions...")
        outputs_with = run_prompt(pair["with"])
        score_with   = measure_consistency(outputs_with)

        results.append({
            "task":          pair["task"],
            "score_without": score_without,
            "score_with":    score_with,
            "improvement":   round(score_with - score_without, 4),
        })

    return results


def print_results(results: list[dict]):
    print("RESULTS SUMMARY")
    print(f"{'Task':<20} {'Without':>10} {'With':>10} {'Improvement':>12}")

    for r in results:
        print(f"{r['task']:<20} {r['score_without']:>10} {r['score_with']:>10} {r['improvement']:>+12}")

    avg_without    = np.mean([r["score_without"] for r in results])
    avg_with       = np.mean([r["score_with"]    for r in results])
    avg_improvement = np.mean([r["improvement"]  for r in results])

    print(f"{'AVERAGE':<20} {avg_without:>10.4f} {avg_with:>10.4f} {avg_improvement:>+12.4f}")
    print(f"\nConclusion: Format instructions {'IMPROVED' if avg_improvement > 0 else 'DID NOT IMPROVE'} consistency by {abs(avg_improvement):.4f} on average.")


if __name__ == "__main__":
    results = run_experiment()
    print_results(results)