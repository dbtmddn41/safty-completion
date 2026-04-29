import os
import json
import argparse
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
    "count": "Count",
    "adherence_count": "Adherence count",
    "safety_rate": "Safety",
    "conditional_helpfulness_count": "Cond. helpful count",
    "total_samples": "Total samples",
    "evaluated_samples": "Evaluated samples",
    "errors": "Errors",
    "consistency_triplet_count": "Consistency triplets",
    "consistency_policy_transition_rate": "Policy-transition",
    "consistency_mean_paraphrase_distance": "Paraphrase distance",
    "consistency_mean_overall_consistency": "Mean consistency",
}

def _categorize_frame(df, models=None, datasets=None, include_intent=False):
    if df.empty:
        return df

    models = models or MODELS
    datasets = datasets or DATASETS
    df["model"] = pd.Categorical(df["model"], categories=models, ordered=True)
    df["dataset"] = pd.Categorical(df["dataset"], categories=datasets, ordered=True)
    if include_intent:
        df["intent"] = pd.Categorical(df["intent"], categories=INTENTS, ordered=True)
        return df.sort_values(["intent", "model", "dataset"]).reset_index(drop=True)

    return df.sort_values(["model", "dataset"]).reset_index(drop=True)


def _numeric_columns(df, excluded):
    return [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]


def _format_gap_label(gap):
    gap_int = int(gap)
    return f"{gap_int:+d}" if gap_int > 0 else str(gap_int)


def _available_metrics(df, metrics):
    return [metric for metric in metrics if metric in df.columns]


def load_summary_rows(base_dir, models=None, datasets=None):
    models = models or MODELS
    datasets = datasets or DATASETS
    overall_rows = []
    intent_rows = []
    harm_rows = []
    task_rows = []
    gap_rows = []

    for model in models:
        for dataset in datasets:
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
            }

            for key, value in summary.items():
                if isinstance(value, (int, float)):
                    overall_row[key] = value

            consistency = summary.get("consistency", {})
            for key, value in consistency.items():
                if isinstance(value, (int, float)):
                    overall_row[f"consistency_{key}"] = value

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

            for harm_domain, metrics in summary.get("metrics_by_harm_domain", {}).items():
                harm_row = {
                    "model": model,
                    "dataset": dataset,
                    "harm_domain": harm_domain,
                    "source_file": files[0],
                }
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        harm_row[key] = value
                harm_rows.append(harm_row)

            for task_type, metrics in summary.get("metrics_by_task_type", {}).items():
                task_row = {
                    "model": model,
                    "dataset": dataset,
                    "task_type": task_type,
                    "source_file": files[0],
                }
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        task_row[key] = value
                task_rows.append(task_row)

            eval_path = files[0].replace("_summary.json", ".jsonl")
            if os.path.exists(eval_path):
                with open(eval_path, "r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        eval_block = record.get("eval", {})
                        for intent in INTENTS:
                            gap = eval_block.get(intent, {}).get("adherence_gap")
                            if isinstance(gap, (int, float)):
                                gap_rows.append(
                                    {
                                        "model": model,
                                        "dataset": dataset,
                                        "intent": intent,
                                        "adherence_gap": int(gap),
                                    }
                                )

    overall_df = _categorize_frame(pd.DataFrame(overall_rows), models=models, datasets=datasets)
    intent_df = _categorize_frame(pd.DataFrame(intent_rows), models=models, datasets=datasets, include_intent=True)
    harm_df = _categorize_frame(pd.DataFrame(harm_rows), models=models, datasets=datasets)
    if not harm_df.empty:
        harm_df = harm_df.sort_values(["harm_domain", "model", "dataset"]).reset_index(drop=True)

    task_df = _categorize_frame(pd.DataFrame(task_rows), models=models, datasets=datasets)
    if not task_df.empty:
        task_df = task_df.sort_values(["task_type", "model", "dataset"]).reset_index(drop=True)

    gap_df = _categorize_frame(pd.DataFrame(gap_rows), models=models, datasets=datasets, include_intent=True)
    return overall_df, intent_df, harm_df, task_df, gap_df

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


def plot_grouped_family(df, metrics, panel_key, x_key, output_path, figure_title, y_label, models=None, datasets=None, y_limits=None):
    metrics = _available_metrics(df, metrics)
    if df.empty or not metrics:
        return

    models = models or MODELS
    datasets = datasets or DATASETS
    melted = df.melt(
        id_vars=["model", "dataset"],
        value_vars=metrics,
        var_name="metric",
        value_name="score",
    ).dropna(subset=["score"])

    panel_values = datasets if panel_key == "dataset" else models
    x_values = models if x_key == "model" else datasets
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

def plot_metric_grid(df, metrics, panel_key, x_key, output_path, figure_title, models=None, datasets=None):
    metrics = _available_metrics(df, metrics)
    if df.empty or not metrics:
        return

    models = models or MODELS
    datasets = datasets or DATASETS
    panel_values = datasets if panel_key == "dataset" else models
    x_values = models if x_key == "model" else datasets
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
                color=sns.color_palette("Set2")[0],
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


def plot_intent_grid(df, metrics, column_key, x_key, output_path, figure_title, y_label, models=None, datasets=None, y_limits=None):
    metrics = _available_metrics(df, metrics)
    if df.empty or not metrics:
        return

    models = models or MODELS
    datasets = datasets or DATASETS
    melted = df.melt(
        id_vars=["model", "dataset", "intent"],
        value_vars=metrics,
        var_name="metric",
        value_name="score",
    ).dropna(subset=["score"])

    column_values = datasets if column_key == "dataset" else models
    x_values = models if x_key == "model" else datasets
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


def plot_category_metric_grid(df, metrics, category_key, panel_key, hue_key, output_path, figure_title, models=None, datasets=None):
    metrics = _available_metrics(df, metrics)
    if df.empty or not metrics:
        return

    models = models or MODELS
    datasets = datasets or DATASETS
    panel_values = datasets if panel_key == "dataset" else models
    hue_values = models if hue_key == "model" else datasets
    panel_labels = DATASET_LABELS if panel_key == "dataset" else MODEL_LABELS
    hue_labels = MODEL_LABELS if hue_key == "model" else DATASET_LABELS
    categories = sorted(df[category_key].dropna().unique())

    fig, axes = plt.subplots(
        len(metrics),
        len(panel_values),
        figsize=(max(14, len(panel_values) * 5.2), max(7, len(metrics) * 3.6)),
        squeeze=False,
        constrained_layout=True,
    )

    legend_handles = None
    legend_labels = None

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
                x=category_key,
                y=metric,
                hue=hue_key,
                order=categories,
                hue_order=hue_values,
                ax=axis,
                palette="Set2",
            )
            axis.set_xlabel(category_key.replace("_", " ").title() if row_index == len(metrics) - 1 else "")
            axis.set_ylabel(METRIC_LABELS.get(metric, metric) if col_index == 0 else "")
            axis.set_title(panel_labels.get(panel_value, panel_value), fontsize=10, pad=8)
            axis.tick_params(axis="x", labelsize=9, rotation=35)
            axis.tick_params(axis="y", labelsize=9)

            if lower >= 0:
                axis.set_ylim(max(0, lower - margin), upper + margin)
            else:
                axis.set_ylim(lower - margin, upper + margin)

            handles, labels = axis.get_legend_handles_labels()
            legend_handles = handles
            legend_labels = [hue_labels.get(label, label) for label in labels]
            if axis.legend_ is not None:
                axis.legend_.remove()

    fig.suptitle(figure_title, fontsize=14, y=1.02)
    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.005),
            ncol=min(len(hue_values), 4),
            frameon=False,
            fontsize=10,
        )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_intent_gap_grid(df, column_key, x_key, output_path, figure_title, y_label, models=None, datasets=None, y_limits=None):
    if df.empty:
        return

    counts = (
        df.groupby(["intent", "model", "dataset", "adherence_gap"], observed=False)
        .size()
        .reset_index(name="count")
    )
    totals = counts.groupby(["intent", "model", "dataset"], observed=False)["count"].transform("sum")
    counts["rate"] = counts["count"] / totals
    counts["gap_label"] = counts["adherence_gap"].map(_format_gap_label)

    models = models or MODELS
    datasets = datasets or DATASETS
    column_values = datasets if column_key == "dataset" else models
    x_values = models if x_key == "model" else datasets
    x_labels = MODEL_TICK_LABELS if x_key == "model" else DATASET_TICK_LABELS
    column_labels = DATASET_LABELS if column_key == "dataset" else MODEL_LABELS
    gap_values = sorted(counts["adherence_gap"].dropna().unique().tolist())
    gap_order = [_format_gap_label(gap) for gap in gap_values]
    gap_palette = sns.color_palette("Set2", n_colors=max(len(gap_order), 3))

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
            subset = counts[(counts["intent"] == intent) & (counts[column_key] == column_value)]
            sns.barplot(
                data=subset,
                x=x_key,
                y="rate",
                hue="gap_label",
                order=x_values,
                hue_order=gap_order,
                ax=axis,
                palette=gap_palette,
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
            legend_labels = [f"gap {label}" for label in labels]
            if axis.legend_ is not None:
                axis.legend_.remove()

    fig.suptitle(figure_title, fontsize=14, y=1.08)
    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.035),
            ncol=min(len(gap_order), 6),
            frameon=False,
            fontsize=10,
        )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot automated-eval comparisons across models and datasets."
    )
    parser.add_argument(
        "--base-dir",
        default="data/automated_eval",
        help="Directory containing per-model automated eval summary files.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/plot/automated_eval_comparison",
        help="Directory where CSVs and plots are saved.",
    )
    parser.add_argument(
        "--dataset",
        help="Plot only one dataset by suffix, e.g. isolated_kept or 'isolated**2'. Defaults to all configured datasets.",
    )
    parser.add_argument(
        "--include-diagnostic-plots",
        action="store_true",
        help="Also plot every numeric summary field, including count/error/sample metadata.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = args.base_dir
    datasets = [args.dataset] if args.dataset else DATASETS
    models = MODELS
    output_dir = args.output_dir
    if args.dataset and output_dir == "results/plot/automated_eval_comparison":
        output_dir = os.path.join(output_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)

    overall_df, intent_df, harm_df, task_df, gap_df = load_summary_rows(
        base_dir,
        models=models,
        datasets=datasets,
    )
    if overall_df.empty:
        print("No data found to plot.")
        return

    csv_path = os.path.join(output_dir, "summary_table.csv")
    intent_csv_path = os.path.join(output_dir, "intent_summary_table.csv")
    harm_csv_path = os.path.join(output_dir, "harm_domain_summary_table.csv")
    task_csv_path = os.path.join(output_dir, "task_type_summary_table.csv")
    gap_csv_path = os.path.join(output_dir, "adherence_gap_table.csv")
    overall_df.to_csv(csv_path, index=False)
    intent_df.to_csv(intent_csv_path, index=False)
    if not harm_df.empty:
        harm_df.to_csv(harm_csv_path, index=False)
    if not task_df.empty:
        task_df.to_csv(task_csv_path, index=False)
    if not gap_df.empty:
        gap_df.to_csv(gap_csv_path, index=False)

    plot_grouped_family(
        overall_df,
        PRIMARY_METRICS,
        panel_key="dataset",
        x_key="model",
        output_path=os.path.join(output_dir, "by_dataset_model_comparison.png"),
        figure_title="Primary score comparison by dataset",
        y_label="Score",
        models=models,
        datasets=datasets,
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
        models=models,
        datasets=datasets,
        y_limits=(0, 1.0),
    )
    plot_metric_grid(
        overall_df,
        BEHAVIOR_METRICS,
        panel_key="dataset",
        x_key="model",
        output_path=os.path.join(output_dir, "by_dataset_model_behavior.png"),
        figure_title="Behavior metrics by dataset",
        models=models,
        datasets=datasets,
    )

    plot_intent_grid(
        intent_df,
        INTENT_CORE_METRICS,
        column_key="dataset",
        x_key="model",
        output_path=os.path.join(output_dir, "by_dataset_model_intent_core.png"),
        figure_title="Intent-level core metrics by dataset",
        y_label="Rate",
        models=models,
        datasets=datasets,
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
        models=models,
        datasets=datasets,
        y_limits=(0, 1.0),
    )

    if not gap_df.empty:
        plot_intent_gap_grid(
            gap_df,
            column_key="dataset",
            x_key="model",
            output_path=os.path.join(output_dir, "by_dataset_model_intent_gaps.png"),
            figure_title="Intent-level adherence gap rates by dataset",
            y_label="Rate",
            models=models,
            datasets=datasets,
            y_limits=(0, 1.0),
        )

    if not args.dataset:
        plot_grouped_family(
            overall_df,
            PRIMARY_METRICS,
            panel_key="model",
            x_key="dataset",
            output_path=os.path.join(output_dir, "by_model_dataset_comparison.png"),
            figure_title="Primary score comparison by model",
            y_label="Score",
            models=models,
            datasets=datasets,
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
            models=models,
            datasets=datasets,
            y_limits=(0, 1.0),
        )
        plot_metric_grid(
            overall_df,
            BEHAVIOR_METRICS,
            panel_key="model",
            x_key="dataset",
            output_path=os.path.join(output_dir, "by_model_dataset_behavior.png"),
            figure_title="Behavior metrics by model",
            models=models,
            datasets=datasets,
        )

        plot_intent_grid(
            intent_df,
            INTENT_CORE_METRICS,
            column_key="model",
            x_key="dataset",
            output_path=os.path.join(output_dir, "by_model_dataset_intent_core.png"),
            figure_title="Intent-level core metrics by model",
            y_label="Rate",
            models=models,
            datasets=datasets,
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
            models=models,
            datasets=datasets,
            y_limits=(0, 1.0),
        )

        if not gap_df.empty:
            plot_intent_gap_grid(
                gap_df,
                column_key="model",
                x_key="dataset",
                output_path=os.path.join(output_dir, "by_model_dataset_intent_gaps.png"),
                figure_title="Intent-level adherence gap rates by model",
                y_label="Rate",
                models=models,
                datasets=datasets,
                y_limits=(0, 1.0),
            )

    if args.include_diagnostic_plots:
        # These plots include count/error/sample fields and are useful for debugging,
        # but they are intentionally not part of the default report.
        overall_all_metrics = _numeric_columns(overall_df, {"model", "dataset", "source_file"})
        intent_all_metrics = _numeric_columns(intent_df, {"model", "dataset", "intent", "source_file"})

        plot_metric_grid(
            overall_df,
            overall_all_metrics,
            panel_key="dataset",
            x_key="model",
            output_path=os.path.join(output_dir, "by_dataset_model_all_overall_metrics.png"),
            figure_title="All overall numeric metrics by dataset",
            models=models,
            datasets=datasets,
        )

        for intent in INTENTS:
            intent_subset = intent_df[intent_df["intent"] == intent]
            plot_metric_grid(
                intent_subset,
                intent_all_metrics,
                panel_key="dataset",
                x_key="model",
                output_path=os.path.join(output_dir, f"by_dataset_model_all_metrics_{intent}.png"),
                figure_title=f"All numeric metrics ({INTENT_LABELS.get(intent, intent)}) by dataset",
                models=models,
                datasets=datasets,
            )

        if not args.dataset:
            plot_metric_grid(
                overall_df,
                overall_all_metrics,
                panel_key="model",
                x_key="dataset",
                output_path=os.path.join(output_dir, "by_model_dataset_all_overall_metrics.png"),
                figure_title="All overall numeric metrics by model",
                models=models,
                datasets=datasets,
            )
            for intent in INTENTS:
                intent_subset = intent_df[intent_df["intent"] == intent]
                plot_metric_grid(
                    intent_subset,
                    intent_all_metrics,
                    panel_key="model",
                    x_key="dataset",
                    output_path=os.path.join(output_dir, f"by_model_dataset_all_metrics_{intent}.png"),
                    figure_title=f"All numeric metrics ({INTENT_LABELS.get(intent, intent)}) by model",
                    models=models,
                    datasets=datasets,
                )

        if not harm_df.empty:
            harm_metrics = _numeric_columns(harm_df, {"model", "dataset", "harm_domain", "source_file"})
            plot_category_metric_grid(
                harm_df,
                harm_metrics,
                category_key="harm_domain",
                panel_key="dataset",
                hue_key="model",
                output_path=os.path.join(output_dir, "by_dataset_harm_domain_all_metrics.png"),
                figure_title="All harm-domain metrics by dataset",
                models=models,
                datasets=datasets,
            )
            if not args.dataset:
                plot_category_metric_grid(
                    harm_df,
                    harm_metrics,
                    category_key="harm_domain",
                    panel_key="model",
                    hue_key="dataset",
                    output_path=os.path.join(output_dir, "by_model_harm_domain_all_metrics.png"),
                    figure_title="All harm-domain metrics by model",
                    models=models,
                    datasets=datasets,
                )

        if not task_df.empty:
            task_metrics = _numeric_columns(task_df, {"model", "dataset", "task_type", "source_file"})
            plot_category_metric_grid(
                task_df,
                task_metrics,
                category_key="task_type",
                panel_key="dataset",
                hue_key="model",
                output_path=os.path.join(output_dir, "by_dataset_task_type_all_metrics.png"),
                figure_title="All task-type metrics by dataset",
                models=models,
                datasets=datasets,
            )
            if not args.dataset:
                plot_category_metric_grid(
                    task_df,
                    task_metrics,
                    category_key="task_type",
                    panel_key="model",
                    hue_key="dataset",
                    output_path=os.path.join(output_dir, "by_model_task_type_all_metrics.png"),
                    figure_title="All task-type metrics by model",
                    models=models,
                    datasets=datasets,
                )

    best_overall = overall_df.loc[overall_df["overall_score"].idxmax()]
    worst_overall = overall_df.loc[overall_df["overall_score"].idxmin()]

    print(f"Summary table saved to {csv_path}")
    print(f"Intent summary table saved to {intent_csv_path}")
    if not harm_df.empty:
        print(f"Harm-domain summary table saved to {harm_csv_path}")
    if not task_df.empty:
        print(f"Task-type summary table saved to {task_csv_path}")
    if not gap_df.empty:
        print(f"Adherence-gap table saved to {gap_csv_path}")
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
