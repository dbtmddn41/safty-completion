import pandas as pd
df = pd.read_csv('results/plot/automated_eval_comparison/summary_table.csv')
metrics = [
    "correct_adherence_rate",
    "over_assistance_rate",
    "mild_over_refusal_rate",
    "unsafe_rate",
    "mean_adherence_gap",
    "mean_conditional_helpfulness",
]
avg = df.groupby('model', observed=True)[metrics].mean()
lower_is_better = ["over_assistance_rate", "mild_over_refusal_rate", "unsafe_rate", "mean_adherence_gap"]
for metric in metrics:
    ascending = metric in lower_is_better
    sorted_avg = avg[metric].sort_values(ascending=ascending)
    print(f"\n{metric}:")
    for model, val in sorted_avg.items():
        print(f"  {model}: {val:.4f}")
