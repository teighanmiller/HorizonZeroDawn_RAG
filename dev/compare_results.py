import json
import sys
from statistics import mean


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_results(results):
    metrics = {
        "Precision": [],
        "Recall": [],
        "F1_Score": [],
        "Cosine Similarity": [],
        "Answer Relevancy": [],
    }

    for entry in results:
        for key in metrics:
            if key in entry:
                metrics[key].append(entry[key])

    # take mean of each metric
    return {key: mean(values) if values else 0 for key, values in metrics.items()}


def compare(file1, file2):
    results1 = load_results(file1)
    results2 = load_results(file2)

    summary1 = summarize_results(results1)
    summary2 = summarize_results(results2)

    print("\n--- Dataset Comparison ---\n")
    print(f"File 1: {file1}")
    print(f"File 2: {file2}\n")

    for metric in summary1.keys():
        val1, val2 = summary1[metric], summary2[metric]
        better = "File 1" if val1 > val2 else ("File 2" if val2 > val1 else "Tie")
        print(f"{metric:20s}: {val1:.4f} vs {val2:.4f} --> {better}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_results.py file1.json file2.json")
    else:
        compare(sys.argv[1], sys.argv[2])
