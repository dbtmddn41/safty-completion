import pandas as pd

df = pd.read_csv("results/plot/automated_eval_comparison/intent_summary_table.csv")

metrics = ["mean_response_score", "correct_adherence_rate", "unsafe_rate"]
intents = ["benign", "dual_use", "malicious"]

# Group by model and intent to get averages across datasets
pivot_df = df.groupby(["model", "intent"])[metrics].mean().reset_index()

# Overall model average across all intents
overall_avg = pivot_df.groupby("model")[metrics].mean().reset_index()

def print_sorted(data, metric, ascending=False):
    sorted_df = data.sort_values(by=metric, ascending=ascending)
    print(f"--- {metric} (Overall Model Average) ---")
    for _, row in sorted_df.iterrows():
        print(f"* {row['model']}: {row[metric]:.4f}")

print_sorted(overall_avg, "mean_response_score", ascending=False)
print_sorted(overall_avg, "correct_adherence_rate", ascending=False)
print_sorted(overall_avg, "unsafe_rate", ascending=True)

# Also show by intent
for intent in intents:
    intent_df = pivot_df[pivot_df["intent"] == intent]
    print(f"\n--- Intent: {intent} ---")
    print_sorted(intent_df, "mean_response_score", ascending=False)
    print_sorted(intent_df, "correct_adherence_rate", ascending=False)
    print_sorted(intent_df, "unsafe_rate", ascending=True)

