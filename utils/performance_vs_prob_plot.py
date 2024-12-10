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
        r"Dev F1 Macro = (\d+\.\d+), Dev F1 Weighted = (\d+\.\d+)"
        r"Dev Precision Macro = (\d+\.\d+), Dev Recall Macro = (\d+\.\d+)"
        r"Dev Accuracy = (\d+\.\d+)"
    )

    with open(log_file, 'r') as f:
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
                    prev_line = next(f, None)
                    metrics_match = metrics_pattern.search(prev_line)
                    if metrics_match:
                        f1_macro, f1_weighted, precision_macro, recall_macro, accuracy = map(float, metrics_match.groups())
                        best_metrics[current_prob] = {
                            "F1 Macro": f1_macro,
                            "F1 Weighted": f1_weighted,
                            "Precision Macro": precision_macro,
                            "Recall Macro": recall_macro,
                            "Accuracy": accuracy,
                        }
                    else:
                        print(f"Metrics not found in line: {prev_line}")
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
    
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot(probabilities, f1_macro_values, marker='o', label="F1 Macro", color="blue")
    plt.plot(probabilities, f1_weighted_values, marker='o', label="F1 Weighted", color="green")
    plt.title("Dev F1 Scores vs. Mask Probability")
    plt.xlabel("Mask Probability")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "f1_scores_vs_mask_probability.png"))
    plt.close()

def plot_individual_metrics(best_metrics, output_dir):
    """
    Plots individual metrics (other than F1 scores) for each mask probability.
    """
    probabilities = sorted(best_metrics.keys())
    metrics = ["Precision Macro", "Recall Macro", "Accuracy"]
    
    os.makedirs(output_dir, exist_ok=True)

    for metric in metrics:
        values = [best_metrics[prob][metric] for prob in probabilities]
        plt.figure(figsize=(8, 6))
        plt.plot(probabilities, values, marker='o', label=metric)
        plt.title(f"Dev {metric} vs. Mask Probability")
        plt.xlabel("Mask Probability")
        plt.ylabel(metric)
        plt.grid()
        plt.savefig(os.path.join(output_dir, f"{metric.replace(' ', '_').lower()}_vs_mask_probability.png"))
        plt.close()

# Paths for logs and plots
log_directory = "../logs"  # Folder containing the log files
plot_directory = "../plots"  # Folder to save the plots

# Parse logs and generate plots
best_metrics = aggregate_metrics(log_directory)
plot_f1_metrics(best_metrics, plot_directory)
plot_individual_metrics(best_metrics, plot_directory)
