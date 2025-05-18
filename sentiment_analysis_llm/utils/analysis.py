from sentiment_analysis.SA_Dataset import SADataset
from transformers import AutoTokenizer

import matplotlib.pyplot as plt

if __name__ == "__main__":
    pretrained_model = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    train_dataset = SADataset('../sa_spaeng/train.conll', tokenizer, mask_out_prob=0)
    eval_dataset = SADataset('../sa_spaeng/dev.conll', tokenizer, mask_out_prob=0)

    stats_train = train_dataset.get_statistics()
    stats_eval = eval_dataset.get_statistics()
    print(
        f"Train dataset statistics: {stats_train}, Eval dataset statistics: {stats_eval}")
    # whole average sentence of train, dev and test
    ave_len = stats_train['avg_sequence_length'] + stats_eval['avg_sequence_length']
    ave_len /= 2
    print(f"Average sentence length: {ave_len}")

    label_dist1 = train_dataset.get_label_distribution()
    label_dist2 = eval_dataset.get_label_distribution()
    print(
        f"Train dataset label distribution: {label_dist1}, Eval dataset label distribution: {label_dist2}")

    # Draw the label distribution, sort by label name
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    # sort
    label_dist1 = dict(sorted(label_dist1.items()))
    label_dist2 = dict(sorted(label_dist2.items()))
    plt.bar(label_dist1.keys(), label_dist1.values())
    plt.title("Train dataset (SPA-ENG) label distribution")
    plt.xticks(rotation=90)
    plt.subplot(1, 2, 2)
    plt.bar(label_dist2.keys(), label_dist2.values())
    plt.title("Eval dataset label (SPA-ENG) distribution")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("../plots/label_distribution.png")
    plt.show()