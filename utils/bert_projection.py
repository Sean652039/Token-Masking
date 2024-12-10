import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from CS_Dataset import CSDataset


def extract_token_embeddings(model, dataset, device):
    """
    Extract token-level embeddings for each class, excluding -100 labels
    """
    model.eval()
    class_embeddings = {0: [], 1: [], 2: []}

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()

            # Get BERT's last hidden state
            outputs = model(inputs, attention_mask=attention_mask, output_hidden_states=True)

            # Last hidden layer's embeddings
            hidden_states = outputs.hidden_states[-1].cpu().numpy()

            # Process each sequence in the batch
            for seq_embedding, seq_labels in zip(hidden_states, labels):
                # Find tokens that are not -100
                valid_mask = (seq_labels != -100)
                valid_labels = seq_labels[valid_mask]
                valid_embeddings = seq_embedding[valid_mask]

                # Group embeddings by their labels
                for label in [0, 1, 2]:
                    label_mask = (valid_labels == label)
                    if np.any(label_mask):
                        # Average embedding for tokens with this label
                        label_embeddings = valid_embeddings[label_mask]
                        class_embeddings[label].append(label_embeddings.mean(axis=0))

    # Convert to numpy arrays
    for label in class_embeddings:
        class_embeddings[label] = np.array(class_embeddings[label])

    return class_embeddings


def plot_class_embeddings(class_embeddings, id2label):
    """
    Plot 3D scatter of class embeddings
    """
    plt.figure(figsize=(12, 10))
    ax = plt.axes(projection='3d')

    # Color map for three classes
    colors = ['red', 'green', 'blue']

    for label, color in zip(range(3), colors):
        # If embeddings exist for this class
        if len(class_embeddings[label]) > 0:
            # Take first 3 dimensions if more than 3
            embeddings = class_embeddings[label][:, :3]

            ax.scatter(
                embeddings[:, 0],
                embeddings[:, 1],
                embeddings[:, 2],
                c=color,
                label=id2label[label],
                alpha=0.7
            )

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.set_title('Token Embeddings by Class')
    plt.legend()

    plt.tight_layout()
    plt.savefig('class_embeddings_3d0.0.png')
    plt.close()  # 关闭图形以节省内存


def main():
    # Configuration
    model_path = "../outputs/20241205_200855/best_model"  # 替换为你最新的模型路径
    test_file = '../lid_spaeng/dev.conll'

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    # Load model and tokenizer
    model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load test dataset
    test_dataset = CSDataset(test_file, tokenizer, mask_out_prob=0)

    # Extract embeddings
    class_embeddings = extract_token_embeddings(model, test_dataset, device)

    # Plot embeddings
    plot_class_embeddings(class_embeddings, test_dataset.id2label)


if __name__ == "__main__":
    main()