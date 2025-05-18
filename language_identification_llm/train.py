from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import get_linear_schedule_with_warmup
import torch
from torch.nn.utils import clip_grad_norm_
from CS_Dataset import CSDataset
from loss import HybridLoss

from sklearn.metrics import f1_score, classification_report, precision_score, recall_score, accuracy_score
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import logging
import os
import random
from datetime import datetime
import json

import sys


def setup_logging(log_dir="logs"):
    """Set up logging configuration"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def set_seed(seed=2731):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_config(config, save_dir):
    """Save training configuration"""
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


def compute_class_weights(dataset):
    """calculate class weights for imbalanced datasets"""
    all_labels = []
    for batch in DataLoader(dataset, batch_size=32):
        labels = batch['labels'].view(-1)  # flatten
        valid_labels = [l.item() for l in labels if l.item() != -100]
        all_labels.extend(valid_labels)

    # calculate class weights
    class_counts = np.bincount(all_labels)

    n_samples = len(all_labels)
    n_classes = len(class_counts)

    weights = n_samples / (n_classes * class_counts + 1e-5)

    # weights = 1 + np.log(weights)
    weights = np.sqrt(weights)

    # Normalize the weights
    weights = weights / weights.sum() * len(weights)

    return torch.FloatTensor(weights)


def compute_metrics(pred_labels, true_labels, attention_mask):
    # only consider the active parts of the sequence
    valid_mask = (attention_mask == 1) & (true_labels != -100)
    active_labels = true_labels[valid_mask]
    active_preds = pred_labels[valid_mask]

    f1_macro = f1_score(active_labels.cpu(), active_preds.cpu(), average='macro', zero_division=0)
    f1_weighted = f1_score(active_labels.cpu(), active_preds.cpu(), average='weighted', zero_division=0)
    precision_macro = precision_score(active_labels.cpu(), active_preds.cpu(), average='macro', zero_division=0)
    recall_macro = recall_score(active_labels.cpu(), active_preds.cpu(), average='macro', zero_division=0)
    accuracy = accuracy_score(active_labels.cpu(), active_preds.cpu())

    # calculate classification report
    report = classification_report(
        active_labels.cpu(),
        active_preds.cpu(),
        output_dict=True,
        zero_division=0
    )

    return {
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'accuracy': accuracy,
        'classification_report': report
    }


def evaluate_model(model, test_dataset, device):
    model.eval()
    dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_predictions = []
    all_labels = []
    all_masks = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(inputs, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)

            all_predictions.append(predictions)
            all_labels.append(labels)
            all_masks.append(attention_mask)

    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    metrics = compute_metrics(all_predictions, all_labels, all_masks)

    return metrics


def train(config, mask_probability):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    """Main training function"""
    logger = setup_logging()
    set_seed(config['seed'])

    # Save configuration
    save_dir = os.path.join("outputs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    save_config(config, save_dir)

    pretrained_model = config['model_name']
    # tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model) # llama

    # Make sure the tokenizer sets the special token
    # special_tokens = {
    #     'pad_token': '<|endoftext|>',  # Qwen可能使用<|endoftext|>作为pad_token
    # }
    special_tokens = {
        'pad_token': '[PAD]',  # Llama doesn't have a pad token by default
    }

    # If the tokenizer does not have specific special tokens set, add them
    tokenizer.add_special_tokens(special_tokens)

    train_dataset = CSDataset('lid_spaeng/train.conll', tokenizer, mask_out_prob=mask_probability)
    eval_dataset = CSDataset('lid_spaeng/dev.conll', tokenizer, mask_out_prob=0)
    test_dataset = CSDataset('lid_nepeng/dev.conll', tokenizer, mask_out_prob=0)

    label2id = train_dataset.label2id
    id2label = train_dataset.id2label
    label_list = list(label2id.keys())

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config['batch_size'], shuffle=False)

    # Use Qwen model and set trust_remote_code=True
    model = AutoModelForTokenClassification.from_pretrained(
        pretrained_model,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        # trust_remote_code=True
    ).to(device)

    # If the tokenizer word list size does not match the model, the word embedding size of the model needs to be adjusted
    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    # add learning rate scheduler
    num_training_steps = len(train_dataloader) * config['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_training_steps // config['warmup_ratio'],
        num_training_steps=num_training_steps
    )

    class_weights = compute_class_weights(train_dataset).to(device)
    print(f"Class Weights: {class_weights}")
    loss_fn = HybridLoss(class_weights, gamma=1.0)

    train_losses = []
    dev_losses = []
    dev_f1_macro = []
    dev_f1_weighted = []
    dev_precision_macro = []
    dev_recall_macro = []
    dev_accuracy = []
    best_f1 = 0

    patience = config['patience']  # early stopping
    min_delta = config['min_delta']
    early_stopping_counter = 0
    best_dev_loss = float('inf')

    model.train()
    logger.info(f"Model: {pretrained_model}")
    logger.info(f"Starting training with mask probability: {mask_probability}")
    for epoch in range(config['num_epochs']):
        epoch_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}"):
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits.view(-1, model.config.num_labels),
                           labels.view(-1))

            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        average_loss = epoch_loss / len(train_dataloader)
        train_losses.append(average_loss)

        # Validation phase
        model.eval()
        epoch_dev_loss = 0
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc=f"Validation Epoch {epoch + 1}/{config['num_epochs']}"):
                inputs = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(inputs, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits.view(-1, model.config.num_labels),
                               labels.view(-1))
                epoch_dev_loss += loss.item()

        average_dev_loss = epoch_dev_loss / len(eval_dataloader)
        dev_losses.append(average_dev_loss)

        # evaluate on the dev set
        dev_metrics = evaluate_model(model, eval_dataset, device)
        dev_f1_macro.append(dev_metrics['f1_macro'])  # F1 Macro
        dev_f1_weighted.append(dev_metrics['f1_weighted'])  # F1 Weighted
        dev_precision_macro.append(dev_metrics['precision_macro'])
        dev_recall_macro.append(dev_metrics['recall_macro'])
        dev_accuracy.append(dev_metrics['accuracy'])

        logger.info(
            f"Epoch {epoch + 1}/{config['num_epochs']}: "
            f"Train Loss = {average_loss:.4f}, "
            f"Dev Loss = {average_dev_loss:.4f}, "
            f"Dev F1 Macro = {dev_metrics['f1_macro']:.4f}, "
            f"Dev F1 Weighted = {dev_metrics['f1_weighted']:.4f}, "
            f"Dev Precision Macro = {dev_metrics['precision_macro']:.4f}, "
            f"Dev Recall Macro = {dev_metrics['recall_macro']:.4f}, "
            f"Dev Accuracy = {dev_metrics['accuracy']:.4f}"
        )

        # Save best model
        if dev_metrics['f1_macro'] > best_f1:
            best_f1 = dev_metrics['f1_macro']
            model_save_path = os.path.join(save_dir, "best_model")
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            logger.info(f"Saved new best model with F1 Macro: {best_f1:.4f}")

        # Early stopping based on dev loss
        if average_dev_loss < best_dev_loss - min_delta:
            best_dev_loss = average_dev_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break

    logger.info("Evaluating on test set...")
    test_metrics = evaluate_model(model, test_dataset, device)
    logger.info(f"Test Results: F1 Macro = {test_metrics['f1_macro']:.4f}, "
                f"F1 Weighted = {test_metrics['f1_weighted']:.4f}")

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Curve')

    plt.subplot(1, 2, 2)
    plt.plot(dev_f1_macro, label='F1 Macro')
    plt.plot(dev_f1_weighted, label='F1 Weighted')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Scores')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"))
    plt.close()

    return {
        'train_losses': train_losses,
        'dev_f1_macro': dev_f1_macro,
        'dev_f1_weighted': dev_f1_weighted,
        'test_metrics': test_metrics,
        'best_f1': best_f1
    }


if __name__ == "__main__":
    config = {
        # 'model_name': "Qwen/Qwen2.5-0.5B-Instruct",  # Modified to Qwen model
        'model_name': "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        'train_file': 'lid_spaeng/train.conll',
        'eval_file': 'lid_spaeng/dev.conll',
        'test_file': 'lid_nepeng/train.conll',
        'batch_size': 64,
        'learning_rate': 1e-5,
        'weight_decay': 0.01,
        'num_epochs': 2000,
        'patience': 10,
        'min_delta': 0.001,
        'warmup_ratio': 0.1,
        'max_grad_norm': 1.0,
        'seed': 2024
    }

    mask_probabilities = [i * 0.1 for i in range(6)]  # 0.0, 0.1, ..., 0.5, can modify as needed
    test_results = {}

    for mask_prob in mask_probabilities:
        print(f"Training with mask probability: {mask_prob}")
        results = train(config, mask_probability=mask_prob)