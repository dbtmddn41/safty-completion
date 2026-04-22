import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

MODELS = ["gemini_pro", "gemini_flash_lite", "llama2"]
DATASETS = ["isolated_kept", "isolated_nc6", "kept_t4096"]
INTENTS = ["benign", "dual_use", "malicious"]

PRIMARY_METRICS = ["overall_score", "response_score", "consistency_score"]
RATE_METRICS = [
    "correct_adherence_rate",
    "over_assistance_rate",
    "mild_over_refusal_rate",
    "unsafe_rate",
]
BEHAVIOR_METRICS = ["mean_adherence_gap", "mean_conditional_helpfulness"]
INTENT_CORE_METRICS = ["mean_response_score", "correct_adherence_rate", "unsafe_rate"]
INTENT_ERROR_METRICS = [
    "over_assistance_rate",
    "mild_over_refusal_rate",
    "severe_over_refusal_rate",
]

MODEL_LABELS = {
    "gemini_pro": "Gemini Pro",
    "gemini_flash_lite": "Gemini Flash Lite",
    "llama2": "Llama 2",
}
MODEL_TICK_LABELS = {
    "gemini_pro": "Gemini\nPro",
    "gemini_flash_lite": "Gemini\nFlash Lite",
    "llama2": "Llama 2",
}
DATASET_LABELS = {
    "isolated_kept": "isolated kept",
    "isolated_nc6": "isolated nc6",
    "kept_t4096": "kept t4096",
}
DATASET_TICK_LABELS = {
    "isolated_kept": "isolated\nkept",
    "isolated_nc6": "isolated\nnc6",
    "kept_t4096": "kept\nt4096",
}
INTENT_LABELS = {
    "benign": "Benign",
    "dual_use": "Dual-use",
    "malicious": "Malicious",
    "overall": "Overall",
}
METRIC_LABELS = {
    "overall_score": "Overall",
    "response_score": "Response",
    "consistency_score": "Consistency",
    "correct_adherence_rate": "Correct adherence",
    "over_assistance_rate": "Over-assistance",
    "mild_over_refusal_rate": "Mild over-refusal",
    "severe_over_refusal_rate": "Severe over-refusal",
    "unsafe_rate": "Unsafe",
    "mean_adherence_gap": "Mean adherence gap",
    "mean_conditional_helpfulness": "Conditional helpfulness",
    "mean_response_score": "Mean response score",
}

def _categorize_frame(df, include_intent=False):
    if df.empty:
        return df

    df["model"] = pd.Categorical(df["model"], categories=MODELS, ordered=True)
    df["dataset"] = pd.Categorical(df["dataset"], categories=DATASETS, ordered=True)
    if include_intent:
        df["intent"] = pd.Categorical(df["intent"], categories=INTENTS, ordered=True)
        return df.sort_values(["intent", "model", "dataset"]).reset_index(drop=True)

    return df.sort_values(["model", "dataset"]).reset_index(drop=True)


def load_summary_rows(base_dir):
    overall_rows = []
    intent_rows = []

    for model in MODELS:
        for dataset in DATASETS:
            file_pattern = os.path.join(base_dir, model, f"*{dataset}_summary.json")
            files = sorted(glob(file_pattern))
            if not files:
                continue

            with open(files[0], "r", encoding="utf-8") as handle:
                summary = json.load(handle)

            overall_intent = summary.get("metrics_by_intent", {}).get("overall", {})
            overall_row = {
                "model": model,
                "dataset": dataset,
                "source_file": files[0],
                "overall_score": summary.get("overall_score"),
                "response_score": summary.get("response_score"),
                "consistency_score": summary.get("consistency_score"),
            }

            for key, value in overall_intent.items():
                if isinstance(value, (int, float)):
                    overall_row[key] = value
            overall_rows.append(overall_row)

            for intent in INTENTS:
                intent_metrics = summary.get("metrics_by_intent", {}).get(intent, {})
                intent_row = {
                    "model": model,
                    "dataset": dataset,
                    "intent": intent,
                    "source_file": files[0],
                }
                for key, value in intent_metrics.items():
                    if isinstance(value, (int, float)):
                        intent_row[key] = value
                intent_rows.append(intent_row)

    overall_df = _categorize_frame(pd.DataFrame(overall_rows))
    intent_df = _categorize_frame(pd.DataFrame(intent_rows), include_intent=True)
    return overall_df, intent_df

def _display_values(values, labels_map):
    return [labels_map.get(value, value) for value in values]

def _style_axis(axis, tick_values, tick_labels, xlabel, ylabel="", y_limits=None, title=None):
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_xticks(range(len(tick_values)))
    axis.set_xticklabels(_display_values(tick_values, tick_labels))
    axis.tick_params(axis="x", labelsize=10)
    axis.tick_params(axis="y", labelsize=10)
    if y_limits is not None:
        axis.set_ylim(*y_limits)
    if title is not None:
        axis.set_title(title, fontsize=11, pad=10)


def plot_grouped_family(df, metrics, panel_key, x_key, output_path, figure_title, y_label, y_limits=None):
    melted = df.melt(
        id_vars=["model", "dataset"],
        value_vars=metrics,
        var_name="metric",
        value_name="score",
    ).dropna(subset=["score"])

    panel_values = DATASETS if panel_key == "dataset" else MODELS
    x_values = MODELS if x_key == "model" else DATASETS
    x_labels = MODEL_TICK_LABELS if x_key == "model" else DATASET_TICK_LABELS
    title_labels = DATASET_LABELS if panel_key == "dataset" else MODEL_LABELS

    fig, axes = plt.subplots(
        1,
        len(panel_values),
        figsize=(max(13, len(panel_values) * 4.6), 5.2),
        sharey=bool(y_limits),
        constrained_layout=True,
    )
    if len(panel_values) == 1:
        axes = [axes]
    legend_handles = None
    legend_labels = None

    for axis_index, (axis, panel_value) in enumerate(zip(axes, panel_values)):
        subset = melted[melted[panel_key] == panel_value]
        sns.barplot(
            data=subset,
            x=x_key,
            y="score",
            hue="metric",
            order=x_values,
            hue_order=metrics,
            ax=axis,
            palette="Set2",
        )
        _style_axis(
            axis,
            x_values,
            x_labels,
            xlabel="Model" if x_key == "model" else "Dataset",
            ylabel=y_label if axis_index == 0 else "",
            y_limits=y_limits,
            title=title_labels.get(panel_value, panel_value),
        )
        handles, labels = axis.get_legend_handles_labels()
        legend_handles = handles
        legend_labels = [METRIC_LABELS.get(label, label) for label in labels]
        if axis.legend_ is not None:
            axis.legend_.remove()

    fig.suptitle(figure_title, fontsize=14, y=1.08)
    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.035),
            ncol=min(len(metrics), 4),
            frameon=False,
            fontsize=10,
        )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_metric_grid(df, metrics, panel_key, x_key, output_path, figure_title):
    panel_values = DATASETS if panel_key == "dataset" else MODELS
    x_values = MODELS if x_key == "model" else DATASETS
    x_labels = MODEL_TICK_LABELS if x_key == "model" else DATASET_TICK_LABELS
    title_labels = DATASET_LABELS if panel_key == "dataset" else MODEL_LABELS

    fig, axes = plt.subplots(
        len(metrics),
        len(panel_values),
        figsize=(max(13, len(panel_values) * 4.6), max(6.5, len(metrics) * 3.2)),
        squeeze=False,
        constrained_layout=True,
    )

    for row_index, metric in enumerate(metrics):
        metric_values = df[metric].dropna()
        if not metric_values.empty:
            lower = metric_values.min()
            upper = metric_values.max()
            margin = max((upper - lower) * 0.15, 0.05)
        else:
            lower, upper, margin = 0, 1, 0.1
        for col_index, panel_value in enumerate(panel_values):
            axis = axes[row_index][col_index]
            subset = df[df[panel_key] == panel_value]
            sns.barplot(
                data=subset,
                x=x_key,
                y=metric,
                order=x_values,
                ax=axis,
                palette="Set2",
            )
            _style_axis(
                axis,
                x_values,
                x_labels,
                xlabel=("Model" if x_key == "model" else "Dataset") if row_index == len(metrics) - 1 else "",
                ylabel=METRIC_LABELS.get(metric, metric) if col_index == 0 else "",
                title=f"{METRIC_LABELS.get(metric, metric)}\n{title_labels.get(panel_value, panel_value)}",
            )
            if lower >= 0:
                axis.set_ylim(max(0, lower - margin), upper + margin)
            else:
                axis.set_ylim(lower - margin, upper + margin)

    fig.suptitle(figure_title, fontsize=14, y=1.04)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_intent_grid(df, metrics, column_key, x_key, output_path, figure_title, y_label, y_limits=None):
    melted = df.melt(
        id_vars=["model", "dataset", "intent"],
        value_vars=metrics,
        var_name="metric",
        value_name="score",
    ).dropna(subset=["score"])

    column_values = DATASETS if column_key == "dataset" else MODELS
    x_values = MODELS if x_key == "model" else DATASETS
    x_labels = MODEL_TICK_LABELS if x_key == "model" else DATASET_TICK_LABELS
    column_labels = DATASET_LABELS if column_key == "dataset" else MODEL_LABELS

    fig, axes = plt.subplots(
        len(INTENTS),
        len(column_values),
        figsize=(max(14, len(column_values) * 4.5), len(INTENTS) * 3.5),
        squeeze=False,
        sharey=bool(y_limits),
        constrained_layout=True,
    )
    legend_handles = None
    legend_labels = None

    for row_index, intent in enumerate(INTENTS):
        for col_index, column_value in enumerate(column_values):
            axis = axes[row_index][col_index]
            subset = melted[(melted["intent"] == intent) & (melted[column_key] == column_value)]
            sns.barplot(
                data=subset,
                x=x_key,
                y="score",
                hue="metric",
                order=x_values,
                hue_order=metrics,
                ax=axis,
                palette="Set2",
            )
            _style_axis(
                axis,
                x_values,
                x_labels,
                xlabel=("Model" if x_key == "model" else "Dataset") if row_index == len(INTENTS) - 1 else "",
                ylabel=y_label if col_index == 0 else "",
                y_limits=y_limits,
                title=f"{INTENT_LABELS.get(intent, intent)}\n{column_labels.get(column_value, column_value)}",
            )
            handles, labels = axis.get_legend_handles_labels()
            legend_handles = handles
            legend_labels = [METRIC_LABELS.get(label, label) for label in labels]
            if axis.legend_ is not None:
                axis.legend_.remove()

    fig.suptitle(figure_title, fontsize=14, y=1.08)
    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.035),
            ncol=min(len(metrics), 4),
            frameon=False,
            fontsize=10,
        )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main():
    base_dir = "data/automated_eval"
    output_dir = "results/plot/automated_eval_comparison"
    os.makedirs(output_dir, exist_ok=True)

    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)

    overall_df, intent_df = load_summary_rows(base_dir)
    if overall_df.empty:
        print("No data found to plot.")
        return

    csv_path = os.path.join(output_dir, "summary_table.csv")
    intent_csv_path = os.path.join(output_dir, "intent_summary_table.csv")
    overall_df.to_csv(csv_path, index=False)
    intent_df.to_csv(intent_csv_path, index=False)

    plot_grouped_family(
        overall_df,
        PRIMARY_METRICS,
        panel_key="dataset",
        x_key="model",
        output_path=os.path.join(output_dir, "by_dataset_model_comparison.png"),
        figure_title="Primary score comparison by dataset",
        y_label="Score",
        y_limits=(0, 1.0),
    )
    plot_grouped_family(
        overall_df,
        PRIMARY_METRICS,
        panel_key="model",
        x_key="dataset",
        output_path=os.path.join(output_dir, "by_model_dataset_comparison.png"),
        figure_title="Primary score comparison by model",
        y_label="Score",
        y_limits=(0, 1.0),
    )
    plot_grouped_family(
        overall_df,
        RATE_METRICS,
        panel_key="dataset",
        x_key="model",
        output_path=os.path.join(output_dir, "by_dataset_model_safety_adherence.png"),
        figure_title="Safety and adherence rates by dataset",
        y_label="Rate",
        y_limits=(0, 1.0),
    )
    plot_grouped_family(
        overall_df,
        RATE_METRICS,
        panel_key="model",
        x_key="dataset",
        output_path=os.path.join(output_dir, "by_model_dataset_safety_adherence.png"),
        figure_title="Safety and adherence rates by model",
        y_label="Rate",
        y_limits=(0, 1.0),
    )
    plot_metric_grid(
        overall_df,
        BEHAVIOR_METRICS,
        panel_key="dataset",
        x_key="model",
        output_path=os.path.join(output_dir, "by_dataset_model_behavior.png"),
        figure_title="Behavior metrics by dataset",
    )
    plot_metric_grid(
        overall_df,
        BEHAVIOR_METRICS,
        panel_key="model",
        x_key="dataset",
        output_path=os.path.join(output_dir, "by_model_dataset_behavior.png"),
        figure_title="Behavior metrics by model",
    )

    plot_intent_grid(
        intent_df,
        INTENT_CORE_METRICS,
        column_key="dataset",
        x_key="model",
        output_path=os.path.join(output_dir, "by_dataset_model_intent_core.png"),
        figure_title="Intent-level core metrics by dataset",
        y_label="Rate",
        y_limits=(0, 1.0),
    )
    plot_intent_grid(
        intent_df,
        INTENT_CORE_METRICS,
        column_key="model",
        x_key="dataset",
        output_path=os.path.join(output_dir, "by_model_dataset_intent_core.png"),
        figure_title="Intent-level core metrics by model",
        y_label="Rate",
        y_limits=(0, 1.0),
    )
    plot_intent_grid(
        intent_df,
        INTENT_ERROR_METRICS,
        column_key="dataset",
        x_key="model",
        output_path=os.path.join(output_dir, "by_dataset_model_intent_errors.png"),
        figure_title="Intent-level error profiles by dataset",
        y_label="Rate",
        y_limits=(0, 1.0),
    )
    plot_intent_grid(
        intent_df,
        INTENT_ERROR_METRICS,
        column_key="model",
        x_key="dataset",
        output_path=os.path.join(output_dir, "by_model_dataset_intent_errors.png"),
        figure_title="Intent-level error profiles by model",
        y_label="Rate",
        y_limits=(0, 1.0),
    )

    best_overall = overall_df.loc[overall_df["overall_score"].idxmax()]
    worst_overall = overall_df.loc[overall_df["overall_score"].idxmin()]

    print(f"Summary table saved to {csv_path}")
    print(f"Intent summary table saved to {intent_csv_path}")
    print(f"Plots saved to {output_dir}/")
    print(
        f"Best overall_score: {best_overall['overall_score']} "
        f"({best_overall['model']} on {best_overall['dataset']})"
    )
    print(
        f"Worst overall_score: {worst_overall['overall_score']} "
        f"({worst_overall['model']} on {worst_overall['dataset']})"
    )

if __name__ == "__main__":
    main()
