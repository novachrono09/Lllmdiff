import json
import os
import asyncio
import re
from pathlib import Path
from typing import Dict, Any, List

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# --- Configuration ---
# Pollinations expects "openai" as the model string.
# Base URL MUST end with /v1 for the proxy to route correctly.
POLLINATIONS_MODEL = "openai"
BASE_URL = "https://gen.pollinations.ai/v1"
CONCURRENCY_LIMIT = 5

# Map of dimension IDs to prompt filenames
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
        # Initialize client with the mandatory /v1 suffix
        self.client = AsyncOpenAI(api_key=api_key, base_url=BASE_URL)
        self.semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
        self.root_path = root_path
        self.prompts_path = root_path / "scorer_prompts"
        self.prompt_cache: Dict[str, str] = {}

    async def load_templates(self):
        """Loads all .txt templates from the scorer_prompts folder."""
        for dim, filename in DIMENSION_MAP.items():
            file_path = self.prompts_path / filename
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    self.prompt_cache[dim] = f.read().strip()
            else:
                print(f"Warning: Template for {dim} not found.")

    def clean_json_text(self, text: str) -> Dict[str, Any]:
        """Robustly extracts and parses JSON from potentially markdown-formatted text."""
        try:
            # Remove potential markdown code blocks
            text = re.sub(r"```json\s*", "", text)
            text = re.sub(r"```\s*", "", text)
            text = text.strip()
            
            # Find the first '{' and last '}' to isolate the JSON object
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                text = text[start:end+1]
                
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return {"score": 0, "reasoning": "JSON parsing failed"}

    async def get_score(self, dimension: str, user_prompt: str, model_response: str, baseline_truth: str = "N/A") -> Dict[str, Any]:
        """Evaluates a single response using only supported Pollinations parameters."""
        template = self.prompt_cache.get(dimension)
        if not template:
            return {"reasoning": f"Missing template for {dimension}", "score": 0}

        try:
            formatted_prompt = template.format(
                user_prompt=user_prompt,
                model_response=model_response,
                baseline_truth=baseline_truth
            )
        except Exception:
            formatted_prompt = f"{template}\n\nUSER PROMPT: {user_prompt}\nMODEL RESPONSE: {model_response}"

        async with self.semaphore:
            try:
                # ONLY pass model, messages, and temperature. Others cause 404s on Pollinations proxy.
                completion = await self.client.chat.completions.create(
                    model=POLLINATIONS_MODEL,
                    messages=[
                        {"role": "system", "content": formatted_prompt + "\nReturn ONLY valid JSON: {\"reasoning\": \"...\", \"score\": <int>}"},
                        {"role": "user", "content": "Evaluate the response."}
                    ],
                    temperature=0.1
                )
                
                content = completion.choices[0].message.content
                return self.clean_json_text(content)
            except Exception as e:
                return {"reasoning": f"API Error: {str(e)}", "score": 0}

    async def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Concurrent evaluation of Model A and Model B."""
        dim = item.get("dimension", "sycophancy")
        u_prompt = item.get("prompt", "")
        b_truth = item.get("baseline_truth", "N/A")

        res_a, res_b = await asyncio.gather(
            self.get_score(dim, u_prompt, item.get("model_a_response", ""), b_truth),
            self.get_score(dim, u_prompt, item.get("model_b_response", ""), b_truth)
        )

        score_a = res_a.get("score", 0)
        score_b = res_b.get("score", 0)
        
        item.update({
            "score_a": score_a,
            "score_b": score_b,
            "distance": abs(score_a - score_b) / 10.0,
            "reasoning_a": res_a.get("reasoning", "Parsing failed"),
            "reasoning_b": res_b.get("reasoning", "Parsing failed")
        })
        return item

async def main():
    api_key = os.environ.get("POLLINATIONS_API_KEY", "any_key")
    root_dir = Path(__file__).parent.parent.absolute()
    input_path = root_dir / "raw_responses.json"
    output_path = root_dir / "scored_responses.json"

    if not input_path.exists():
        print(f"ERROR: {input_path} not found.")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scorer = BehavioralScorer(api_key, root_dir)
    await scorer.load_templates()

    print(f"Scoring {len(data)} items via Pollinations...")
    tasks = [scorer.process_item(item) for item in data]
    scored_results = await tqdm.gather(*tasks, desc="Judging")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scored_results, f, indent=4, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
