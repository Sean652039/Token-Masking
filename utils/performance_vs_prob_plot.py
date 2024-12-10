import re
import matplotlib.pyplot as plt
import os


def parse_logs(log_file):
    """
    Parses a single log file to extract the best metrics for the corresponding mask probability.
    """
    best_metrics = {}
    f1_macro_pattern = re.compile(r"F1 Macro: (\d+\.\d+)")
    metrics_pattern = re.compile(
        r"Dev Loss = (\d+\.\d+), Dev F1 Macro = (\d+\.\d+), Dev F1 Weighted = (\d+\.\d+)"
        r"Dev Precision Macro = (\d+\.\d+), Dev Recall Macro = (\d+\.\d+)"
        r"Dev Accuracy = (\d+\.\d+)"
    )
    test_metrics_pattern = re.compile(r"F1 Macro = (\d+\.\d+), F1 Weighted = (\d+\.\d+)")

    with open(log_file, 'r') as f:
        prev_line = None
        for line in f:
            # Detect the mask probability
            if "Starting training with mask probability" in line:
                current_prob = float(re.search(r"mask probability: (\d+\.\d+)", line).group(1))
            
            # Detect metrics for the current epoch
            if "Saved new best model" in line:
                f1_macro_match = f1_macro_pattern.search(line)
                if f1_macro_match:
                    f1_macro = float(f1_macro_match.group(1))
                    
                    # Extract additional metrics from the preceding line
                    # prev_line = next(f, None)
                    metrics_match = metrics_pattern.search(prev_line)
                    if metrics_match:
                        dev_loss, f1_macro, f1_weighted, precision_macro, recall_macro, accuracy = map(float, metrics_match.groups())
                        best_metrics[current_prob] = {
                            "Dev Loss": dev_loss,
                            "F1 Macro": f1_macro,
                            "F1 Weighted": f1_weighted,
                            "Precision Macro": precision_macro,
                            "Recall Macro": recall_macro,
                            "Accuracy": accuracy,
                        }
                    else:
                        print(f"Metrics not found in line: {prev_line}")
            prev_line = line
        last_line = prev_line
        test_metricss_match = test_metrics_pattern.search(last_line)
        if test_metricss_match:
            f1_macro, f1_weighted = map(float, test_metricss_match.groups())
            test_metrics = {
                "Test F1 Macro": f1_macro,
                "Test F1 Weighted": f1_weighted,
            }
            best_metrics[current_prob].update(test_metrics)

    return best_metrics


def aggregate_metrics(log_dir):
    """
    Aggregates metrics from all log files in the specified directory.
    """
    aggregated_metrics = {}
    for log_file in os.listdir(log_dir):
        if log_file.endswith(".log"):  # Process only log files
            full_path = os.path.join(log_dir, log_file)
            file_metrics = parse_logs(full_path)
            aggregated_metrics.update(file_metrics)
    return aggregated_metrics

def plot_f1_metrics(best_metrics, output_dir):
    """
    Plots F1 Macro and F1 Weighted metrics on the same plot.
    """
    probabilities = sorted(best_metrics.keys())
    f1_macro_values = [best_metrics[prob]["F1 Macro"] for prob in probabilities]
    f1_weighted_values = [best_metrics[prob]["F1 Weighted"] for prob in probabilities]

    test_f1_macro_values = [best_metrics[prob]["Test F1 Macro"] for prob in probabilities]
    test_f1_weighted_values = [best_metrics[prob]["Test F1 Weighted"] for prob in probabilities]
    
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update({
        'font.family': 'serif',  # 使用衬线字体
        'font.serif': ['Times New Roman'],  # Times New Roman 字体
        'axes.titlesize': 14,  # 标题大小
        'axes.labelsize': 12,  # 坐标轴标签大小
        'xtick.labelsize': 10,  # x轴标签大小
        'ytick.labelsize': 10,  # y轴标签大小
        'legend.fontsize': 10,  # 图例字体大小
        'axes.linewidth': 1.5,  # 坐标轴线宽
        'grid.alpha': 0.3  # 网格透明度
    })

    plt.figure(figsize=(8, 6))
    plt.plot(probabilities, f1_macro_values, marker='o', label="F1 Macro", color="blue", markersize=8, linestyle='-',
             linewidth=2)
    plt.plot(probabilities, f1_weighted_values, marker='s', label="F1 Weighted", color="green", markersize=8,
             linestyle='-', linewidth=2)

    plt.axhline(y=f1_macro_values[0], color='red', linestyle='--', linewidth=1, label="Mask Probability = 0.0")
    plt.axhline(y=f1_weighted_values[0], color='purple', linestyle='--', linewidth=1, label="Mask Probability = 0.0")

    plt.title("Dev F1 Scores vs. Mask Probability", fontsize=16)
    plt.xlabel("Mask Probability", fontsize=14)
    plt.ylabel("F1 Score", fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='upper right', frameon=False)
    plt.savefig(os.path.join(output_dir, "f1_scores_vs_mask_probability.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(probabilities, test_f1_macro_values, marker='o', label="Test F1 Macro", color="blue", markersize=8, linestyle='-',
             linewidth=2)
    plt.plot(probabilities, test_f1_weighted_values, marker='s', label="Test F1 Weighted", color="green", markersize=8,
                linestyle='-', linewidth=2)
    plt.axhline(y=test_f1_macro_values[0], color='red', linestyle='--', linewidth=1, label="Mask Probability = 0.0")
    plt.axhline(y=test_f1_weighted_values[0], color='purple', linestyle='--', linewidth=1, label="Mask Probability = 0.0")
    plt.title("Test F1 Scores vs. Mask Probability", fontsize=16)
    plt.xlabel("Mask Probability", fontsize=14)
    plt.ylabel("F1 Score", fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='upper right', frameon=False)
    plt.savefig(os.path.join(output_dir, "test_f1_scores_vs_mask_probability.png"))
    plt.close()

def plot_individual_metrics(best_metrics, output_dir):
    """
    Plots individual metrics (other than F1 scores) for each mask probability.
    """
    probabilities = sorted(best_metrics.keys())
    metrics = ["Precision Macro", "Recall Macro", "Accuracy"]
    
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update({
        'font.family': 'serif',  # 使用衬线字体
        'font.serif': ['Times New Roman'],  # Times New Roman 字体
        'axes.titlesize': 14,  # 标题大小
        'axes.labelsize': 12,  # 坐标轴标签大小
        'xtick.labelsize': 10,  # x轴标签大小
        'ytick.labelsize': 10,  # y轴标签大小
        'legend.fontsize': 10,  # 图例字体大小
        'axes.linewidth': 1.5,  # 坐标轴线宽
        'grid.alpha': 0.3  # 网格透明度
    })

    for metric in metrics:
        values = [best_metrics[prob][metric] for prob in probabilities]
        plt.figure(figsize=(8, 6))
        plt.plot(probabilities, values, marker='o', label=metric, markersize=8, linestyle='-', linewidth=2, color='blue')
        plt.axhline(y=values[0], color='red', linestyle='--', linewidth=1, label="Mask Probability = 0.0")
        plt.title(f"Dev {metric} vs. Mask Probability", fontsize=16)
        plt.xlabel("Mask Probability", fontsize=14)
        plt.ylabel(metric, fontsize=14)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(loc='upper right', frameon=False)
        plt.savefig(os.path.join(output_dir, f"{metric.replace(' ', '_').lower()}_vs_mask_probability.png"))
        plt.close()

# Paths for logs and plots
log_directory = "../logs"  # Folder containing the log files
plot_directory = "../plots"  # Folder to save the plots

# Parse logs and generate plots
best_metrics = aggregate_metrics(log_directory)
plot_f1_metrics(best_metrics, plot_directory)
plot_individual_metrics(best_metrics, plot_directory)
# Sort the best metrics by mask probability
sorted_metrics = {k: best_metrics[k] for k in sorted(best_metrics)}
print(sorted_metrics)
