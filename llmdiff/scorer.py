import json
import os
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional

from groq import AsyncGroq
from tqdm.asyncio import tqdm

# --- Configuration ---
GROQ_MODEL = "qwen-3-32b"  # Latest Qwen model on Groq
CONCURRENCY_LIMIT = 5

# Map of dimension IDs (as they appear in battery.json) to prompt filenames
DIMENSION_MAP = {
    "sycophancy": "sycophancy.txt",
    "refusal_rate": "refusal_rate.txt",
    "hallucination": "hallucination.txt",
    "confidence_calibration": "confidence_calibration.txt",
    "reasoning_style": "reasoning_style.txt",
    "verbosity_caveat_bloat": "verbosity_caveat_bloat.txt"
}

class BehavioralScorer:
    def __init__(self, api_key: str, root_path: Path):
        self.client = AsyncGroq(api_key=api_key)
        self.semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
        self.root_path = root_path
        self.prompts_path = root_path / "scorer_prompts"
        self.prompt_cache: Dict[str, str] = {}

    async def load_templates(self):
        """Loads all .txt templates from the scorer_prompts folder into memory."""
        for dim, filename in DIMENSION_MAP.items():
            file_path = self.prompts_path / filename
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    self.prompt_cache[dim] = f.read().strip()
            else:
                print(f"Warning: Template for {dim} not found at {file_path}")

    async def get_score(self, dimension: str, user_prompt: str, model_response: str, baseline_truth: str = "N/A") -> Dict[str, Any]:
        """Evaluates a single response using the Groq API."""
        template = self.prompt_cache.get(dimension)
        if not template:
            return {"reasoning": f"Missing template for {dimension}", "score": 0}

        # Safe formatting of the template
        try:
            formatted_prompt = template.format(
                user_prompt=user_prompt,
                model_response=model_response,
                baseline_truth=baseline_truth
            )
        except Exception as e:
            # Fallback if template uses different keys or formatting fails
            formatted_prompt = f"{template}\n\nUSER PROMPT: {user_prompt}\nMODEL RESPONSE: {model_response}"

        async with self.semaphore:
            try:
                completion = await self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": formatted_prompt},
                        {"role": "user", "content": "Return the JSON evaluation now."}
                    ],
                    model=GROQ_MODEL,
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                
                result = json.loads(completion.choices[0].message.content)
                # Ensure keys exist and are of correct type
                return {
                    "reasoning": str(result.get("reasoning", "No reasoning provided")),
                    "score": int(result.get("score", 0))
                }
            except Exception as e:
                return {"reasoning": f"Evaluation Error: {str(e)}", "score": 0}

    async def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrates the dual evaluation of Model A and Model B."""
        dim = item.get("dimension", "sycophancy")
        u_prompt = item.get("prompt", "")
        b_truth = item.get("baseline_truth", "N/A")

        # Evaluate both models concurrently
        eval_a_task = self.get_score(dim, u_prompt, item.get("model_a_response", ""), b_truth)
        eval_b_task = self.get_score(dim, u_prompt, item.get("model_b_response", ""), b_truth)
        
        res_a, res_b = await asyncio.gather(eval_a_task, eval_b_task)

        score_a = res_a["score"]
        score_b = res_b["score"]
        distance = abs(score_a - score_b) / 10.0

        item.update({
            "score_a": score_a,
            "score_b": score_b,
            "distance": distance,
            "reasoning_a": res_a["reasoning"],
            "reasoning_b": res_b["reasoning"]
        })
        return item

async def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("CRITICAL: GROQ_API_KEY not found in environment.")
        return

    # Resolve paths relative to this script's location (llmdiff/scorer.py -> root/)
    root_dir = Path(__file__).parent.parent.absolute()
    input_path = root_dir / "raw_responses.json"
    output_path = root_dir / "scored_responses.json"

    if not input_path.exists():
        print(f"ERROR: Input file {input_path} not found.")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scorer = BehavioralScorer(api_key, root_dir)
    await scorer.load_templates()

    print(f"Starting behavioral evaluation of {len(data)} items...")
    
    tasks = [scorer.process_item(item) for item in data]
    scored_results = await tqdm.gather(*tasks, desc="Judging Divergence")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scored_results, f, indent=4, ensure_ascii=False)

    print(f"\nSuccess! Scored results saved to: {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
