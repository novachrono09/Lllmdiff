import json
import os
import asyncio
import re
from typing import Dict, Any, List

from groq import AsyncGroq
from tqdm.asyncio import tqdm

# --- Configuration ---
GROQ_MODEL = "qwen-2.5-32b"  # Latest Qwen model on Groq as of now
CONCURRENCY_LIMIT = 5
INPUT_FILE = "raw_responses.json"
OUTPUT_FILE = "scored_responses.json"
PROMPT_DIR = "scorer_prompts"

# --- Behavioral Rubrics ---
# We prioritize loading from the scorer_prompts directory created earlier.
# If a file is missing, we use these placeholders.
SYSTEM_PROMPTS = {
    "sycophancy": "Evaluate the response for sycophancy. (Placeholder)",
    "refusal_rate": "Evaluate the response for refusal rate. (Placeholder)",
    "hallucination": "Evaluate the response for hallucination. (Placeholder)",
    "confidence_calibration": "Evaluate the response for confidence calibration. (Placeholder)",
    "reasoning_style": "Evaluate the response for reasoning style. (Placeholder)",
    "verbosity_caveat_bloat": "Evaluate the response for verbosity. (Placeholder)",
}

def load_prompts():
    """Load rubrics from .txt files in scorer_prompts directory if they exist."""
    if not os.path.exists(PROMPT_DIR):
        return
    
    for filename in os.listdir(PROMPT_DIR):
        if filename.endswith(".txt"):
            dimension = filename.replace(".txt", "")
            with open(os.path.join(PROMPT_DIR, filename), "r") as f:
                SYSTEM_PROMPTS[dimension] = f.read().strip()

class BehavioralJudge:
    def __init__(self, api_key: str):
        self.client = AsyncGroq(api_key=api_key)
        self.semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def score_response(self, dimension: str, prompt: str, response: str) -> Dict[str, Any]:
        """Evaluates a single model response against a behavioral dimension."""
        system_msg = SYSTEM_PROMPTS.get(dimension, SYSTEM_PROMPTS["sycophancy"])
        
        # Ensure JSON constraint is part of the system prompt
        json_instruction = '\nOutput ONLY valid JSON in the format: {"reasoning": "<short text>", "score": <int 0-10>}'
        
        # Format the system prompt with context if placeholders are present
        # Note: Our .txt prompts already contain placeholders like {user_prompt}
        try:
            formatted_system = system_msg.format(
                user_prompt=prompt,
                model_response=response,
                baseline_truth="N/A (Standard Behavioral Evaluation)"
            )
        except KeyError:
            # Fallback for simple placeholders or if .format fails
            formatted_system = f"{system_msg}\n\nUSER PROMPT: {prompt}\nMODEL RESPONSE: {response}"

        async with self.semaphore:
            try:
                chat_completion = await self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": formatted_system + json_instruction},
                        {"role": "user", "content": "Please evaluate the provided response."}
                    ],
                    model=GROQ_MODEL,
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                
                content = chat_completion.choices[0].message.content
                return json.loads(content)
            except Exception as e:
                return {"reasoning": f"Error: {str(e)}", "score": 0}

async def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY environment variable not set.")
        return

    load_prompts()
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    judge = BehavioralJudge(api_key)
    scored_data = []

    print(f"Scoring {len(data)} items using {GROQ_MODEL}...")

    async def process_item(item):
        dim = item["dimension"]
        prompt = item["prompt"]
        
        # Score Model A and Model B independently
        res_a, res_b = await asyncio.gather(
            judge.score_response(dim, prompt, item["model_a_response"]),
            judge.score_response(dim, prompt, item["model_b_response"])
        )

        score_a = res_a.get("score", 0)
        score_b = res_b.get("score", 0)
        distance = abs(score_a - score_b) / 10.0

        item.update({
            "score_a": score_a,
            "score_b": score_b,
            "distance": distance,
            "reasoning_a": res_a.get("reasoning", ""),
            "reasoning_b": res_b.get("reasoning", "")
        })
        return item

    # Use tqdm for async processing
    tasks = [process_item(item) for item in data]
    scored_data = await tqdm.gather(*tasks, desc="Judging Divergence")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(scored_data, f, indent=4)

    print(f"Successfully saved results to {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
