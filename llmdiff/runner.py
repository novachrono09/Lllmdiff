import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

def main():
    # Configuration
    input_file = "battery.json"
    output_file = "raw_responses.json"
    model_a_id = "Qwen/Qwen2.5-0.5B"
    model_b_id = "Qwen/Qwen2.5-0.5B-Instruct"

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"'{input_file}' not found. Please provide the input dataset.")

    with open(input_file, "r", encoding="utf-8") as f:
        battery = json.load(f)[:12]  # Fast Dev Mode: Only process first 12 prompts

    print("Configuring 4-bit quantization for T4 GPUs...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # --- Load Model A (Base) on cuda:0 ---
    print(f"Loading Model A ({model_a_id}) on cuda:0...")
    tokenizer_a = AutoTokenizer.from_pretrained(model_a_id)
    model_a = AutoModelForCausalLM.from_pretrained(
        model_a_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
    )

    # --- Load Model B (Instruct) on cuda:1 ---
    print(f"Loading Model B ({model_b_id}) on cuda:1...")
    tokenizer_b = AutoTokenizer.from_pretrained(model_b_id)
    model_b = AutoModelForCausalLM.from_pretrained(
        model_b_id,
        quantization_config=bnb_config,
        device_map={"": 1},
        torch_dtype=torch.float16,
    )

    results = []

    print(f"Processing {len(battery)} prompts...")
    for item in tqdm(battery, desc="Generating Responses"):
        prompt = item["prompt"]

        # --- Generate Response for Model A (Base) ---
        # Base models take the raw string directly
        inputs_a = tokenizer_a(prompt, return_tensors="pt").to("cuda:0")
        
        with torch.no_grad():
            outputs_a = model_a.generate(
                **inputs_a,
                max_new_tokens=150,
                do_sample=False,  # Greedy decoding for consistent comparison
                pad_token_id=tokenizer_a.eos_token_id
            )
        
        # Extract only the newly generated tokens
        input_length_a = inputs_a.input_ids.shape[1]
        response_a = tokenizer_a.decode(outputs_a[0][input_length_a:], skip_special_tokens=True).strip()

        # --- Generate Response for Model B (Instruct) ---
        # Instruct models require their specific chat template
        messages = [{"role": "user", "content": prompt}]
        prompt_b = tokenizer_b.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs_b = tokenizer_b(prompt_b, return_tensors="pt").to("cuda:1")

        with torch.no_grad():
            outputs_b = model_b.generate(
                **inputs_b,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer_b.eos_token_id
            )
        
        # Extract only the newly generated tokens
        input_length_b = inputs_b.input_ids.shape[1]
        response_b = tokenizer_b.decode(outputs_b[0][input_length_b:], skip_special_tokens=True).strip()

        # --- Store Result ---
        results.append({
            "id": item.get("id"),
            "dimension": item.get("dimension"),
            "prompt": prompt,
            "model_a_response": response_a,
            "model_b_response": response_b
        })

    print(f"Saving results to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    print("Done!")

if __name__ == "__main__":
    main()
