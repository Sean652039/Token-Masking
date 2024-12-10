import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from CS_Dataset import CSDataset


def extract_logits(model, dataset, device):
    """
    Extract logits for each class, excluding -100 labels
    """
    model.eval()
    class_logits = {0: [], 1: [], 2: []}

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()

            # Get model's logits
            outputs = model(inputs, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy()

            # Process each sequence in the batch
            for seq_logits, seq_labels in zip(logits, labels):
                # Find tokens that are not -100
                valid_mask = (seq_labels != -100)
                valid_labels = seq_labels[valid_mask]
                valid_seq_logits = seq_logits[valid_mask]

                # Group logits by their labels
                for label in [0, 1, 2]:
                    label_mask = (valid_labels == label)
                    if np.any(label_mask):
                        # Average logits for tokens with this label
                        label_logits = valid_seq_logits[label_mask]
                        class_logits[label].append(label_logits.mean(axis=0))

    # Convert to numpy arrays
    for label in class_logits:
        class_logits[label] = np.array(class_logits[label])

    return class_logits


def compute_axis_limits(all_models_logits):
    """
    Compute consistent axis limits for all subplots
    """
    all_values = []
    for model_logits in all_models_logits.values():
        for class_logits in model_logits.values():
            all_values.extend(class_logits.flatten())

    # Add some padding
    min_val = np.min(all_values)
    max_val = np.max(all_values)
    padding = (max_val - min_val) * 0.1

    return min_val - padding, max_val + padding


def plot_multi_model_logits(all_models_logits, id2label):
    """
    Plot logits distribution for multiple models from different perspectives
    """
    # Compute consistent axis limits
    x_lim = compute_axis_limits(all_models_logits)

    # 3D Plot
    fig_3d = plt.figure(figsize=(15, 10), dpi=300)
    plt.subplots_adjust(wspace=0.25, hspace=0.1)

    for col, (mask_prob, class_logits) in enumerate(all_models_logits.items()):
        ax = fig_3d.add_subplot(2, 3, col + 1, projection='3d')

        # Publication-friendly color palette
        colors = ['#1F77B4', '#FF7F0E', '#2CA02C']
        markers = ['o', '^', 's']

        for label_idx, label_name in enumerate(range(3)):
            if len(class_logits[label_name]) > 0:
                logits = class_logits[label_name]

                ax.scatter(
                    logits[:, 0], logits[:, 1], logits[:, 2],
                    c=colors[label_idx],
                    marker=markers[label_idx],
                    label=id2label[label_idx],
                    alpha=0.7,
                    edgecolors='black',
                    linewidths=0.5,
                    s=30
                )

        # Set consistent axis limits
        ax.set_xlim(x_lim)
        ax.set_ylim(x_lim)
        ax.set_zlim(x_lim)

        ax.set_xlabel('Logits D0', fontsize=8, fontweight='bold')
        ax.set_ylabel('Logits D1', fontsize=8, fontweight='bold')
        ax.set_zlabel('Logits D2', fontsize=8, fontweight='bold')
        ax.set_title(f'Mask Out Probability: {mask_prob:.2f}', fontsize=10, fontweight='bold')

        ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
        ax.legend(title='Classes', loc='best', title_fontsize=8, fontsize=7)

    plt.suptitle('3D Logits Distribution Across Different Mask Probabilities', fontsize=16, fontweight='bold')
    plt.savefig('../plots/logits_distribution_3d.png', bbox_inches='tight')
    plt.close()

    # 2D Plots
    perspectives = [
        {'dims': [0, 1], 'title': 'Dim 0 vs Dim 1'},
        {'dims': [1, 2], 'title': 'Dim 1 vs Dim 2'}
    ]

    for perspective in perspectives:
        fig_2d = plt.figure(figsize=(15, 10), dpi=300)
        plt.subplots_adjust(wspace=0.1, hspace=0.2)

        for col, (mask_prob, class_logits) in enumerate(all_models_logits.items()):
            ax = fig_2d.add_subplot(2, 3, col + 1)

            # Publication-friendly color palette
            colors = ['#1F77B4', '#FF7F0E', '#2CA02C']
            markers = ['o', '^', 's']

            x_dim, y_dim = perspective['dims']

            for label_idx, label_name in enumerate(range(3)):
                if len(class_logits[label_name]) > 0:
                    logits = class_logits[label_name]

                    ax.scatter(
                        logits[:, x_dim], logits[:, y_dim],
                        c=colors[label_idx],
                        marker=markers[label_idx],
                        label=id2label[label_idx],
                        alpha=0.7,
                        edgecolors='black',
                        linewidths=0.5,
                        s=30
                    )

            # Set consistent axis limits
            ax.set_xlim(x_lim)
            ax.set_ylim(x_lim)

            ax.set_xlabel(f'Logits D{x_dim}', fontsize=8, fontweight='bold')
            ax.set_ylabel(f'Logits D{y_dim}', fontsize=8, fontweight='bold')
            ax.set_title(f'Mask Out Probability: {mask_prob:.2f}', fontsize=10, fontweight='bold')

            ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
            ax.legend(title='Classes', loc='best', title_fontsize=8, fontsize=7)

        plt.suptitle(f'2D Logits Distribution: {perspective["title"]} Across Different Mask Probabilities', fontsize=16,
                     fontweight='bold')
        plt.savefig(f'../plots/logits_distribution_{x_dim}_{y_dim}.png', bbox_inches='tight')
        plt.close()


def main():
    # Configuration
    outputs_dir = "../outputs"
    test_file = '../lid_spaeng/dev.conll'

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    # Collect all models
    all_models_logits = {}

    # Sort model directories to ensure consistent order
    model_dirs = sorted([d for d in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, d))])

    mask_prob = 0.0
    for model_dir in model_dirs:
        model_path = os.path.join(outputs_dir, model_dir, "best_model")

        # Load model and tokenizer
        model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load test dataset with current mask probability
        test_dataset = CSDataset(test_file, tokenizer, mask_out_prob=0)

        # Extract logits
        class_logits = extract_logits(model, test_dataset, device)

        # Store logits with mask probability as key
        all_models_logits[mask_prob] = class_logits

        mask_prob += 0.1

    # Plot logits for all models
    plot_multi_model_logits(all_models_logits, test_dataset.id2label)


if __name__ == "__main__":
    main()