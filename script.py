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
avg = df.groupby('model')[metrics].mean()
for metric in metrics:
    best_model = avg[metric].idxmax() if metric not in ["over_assistance_rate", "mild_over_refusal_rate", "unsafe_rate", "mean_adherence_gap"] else avg[metric].idxmin()
    worst_model = avg[metric].idxmin() if metric not in ["over_assistance_rate", "mild_over_refusal_rate", "unsafe_rate", "mean_adherence_gap"] else avg[metric].idxmax()
    print(f"{metric}: Best: {best_model} ({avg.loc[best_model, metric]:.4f}), Worst: {worst_model} ({avg.loc[worst_model, metric]:.4f})")
