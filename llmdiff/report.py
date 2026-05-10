import json
from pathlib import Path
from collections import defaultdict

def main():
    # Resolve paths relative to this script
    root_dir = Path(__file__).parent.parent.absolute()
    input_path = root_dir / "scored_responses.json"
    output_path = root_dir / "summary_report.json"

    if not input_path.exists():
        print(f"Error: Scored results not found at {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        print("Error: scored_responses.json is empty.")
        return

    # Initialize aggregators
    dimensions_data = defaultdict(lambda: {"score_a": [], "score_b": [], "distances": []})
    total_distances = []

    # Aggregate scores and distances
    for item in data:
        dim = item.get("dimension", "unknown")
        score_a = item.get("score_a", 0)
        score_b = item.get("score_b", 0)
        dist = item.get("distance", 0.0)

        dimensions_data[dim]["score_a"].append(score_a)
        dimensions_data[dim]["score_b"].append(score_b)
        dimensions_data[dim]["distances"].append(dist)
        total_distances.append(dist)

    # Calculate global fingerprint
    global_fingerprint = sum(total_distances) / len(total_distances) if total_distances else 0.0

    # Calculate per-dimension stats
    summary = {
        "behavioral_fingerprint": round(global_fingerprint, 4),
        "dimensions": {}
    }

    for dim, values in dimensions_data.items():
        avg_a = (sum(values["score_a"]) / len(values["score_a"])) / 10.0
        avg_b = (sum(values["score_b"]) / len(values["score_b"])) / 10.0
        
        summary["dimensions"][dim] = {
            "model_a": round(avg_a, 4),
            "model_b": round(avg_b, 4),
            "distance": round(abs(avg_a - avg_b), 4)
        }

    # Save to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    # --- Pretty Terminal Output ---
    print("\n" + "="*50)
    print("       LLM BEHAVIORAL DIVERGENCE REPORT")
    print("="*50)
    print(f"Global Behavioral Fingerprint: {summary['behavioral_fingerprint']:.4f}")
    print("-"*50)
    print(f"{'Dimension':<25} | {'Model A':<8} | {'Model B':<8} | {'Diff':<8}")
    print("-"*50)
    
    # Sort dimensions for consistent output
    for dim in sorted(summary["dimensions"].keys()):
        stats = summary["dimensions"][dim]
        print(f"{dim:<25} | {stats['model_a']:<8.4f} | {stats['model_b']:<8.4f} | {stats['distance']:<8.4f}")
    
    print("="*50)
    print(f"Full report saved to: {output_path}\n")

if __name__ == "__main__":
    main()
