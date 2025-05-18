from torch.utils.data import Dataset
import torch
import numpy as np
from typing import List, Dict
import logging
import os


class CSDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=15, mask_out_prob=0.15, label_pad_token_id: int = -100):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_out_prob = mask_out_prob
        self.label_pad_token_id = label_pad_token_id

        self.sentences, self.labels, all_labels = self._read_conll_file(file_path)

        # label2id = {label: idx for idx, label in enumerate(sorted(set(all_labels)))}
        # # sort by label name
        # self.label2id = dict(sorted(label2id.items()))
        # self.id2label = {idx: label for label, idx in self.label2id.items()}
        self.label2id = {"lang1": 0, "lang2": 1, "other": 2}
        self.id2label = {0: "lang1", 1: "lang2", 2: "other"}

        self.encoded_data = self._preprocess_data()

        # # Set up logging
        # self.logger = logging.getLogger(__name__)

        # hint: cache
        # self.cache_dir = cache_dir

    def _read_conll_file(self, file_path: str) -> tuple[List[List[str]], List[List[str]], List[str]]:
        """
        Read a CoNLL file and return sentences and labels.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        sentences, labels = [], []
        current_sentence, current_labels = [], []
        all_labels = set()

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '#' in line:
                    continue
                line = line.strip()

                if line:
                    # Split the line by tab or space
                    parts = line.split('\t') if '\t' in line else line.split()

                    if len(parts) >= 2:  # make sure the line has at least two columns
                        token, label = parts[0], parts[-1]
                        if label not in ['lang1', 'lang2', 'other']:
                            label = 'other'
                        current_sentence.append(token)
                        current_labels.append(label)
                        all_labels.add(label)
                elif current_sentence:  # empty line marks the end of a sentence
                    sentences.append(current_sentence)
                    labels.append(current_labels)
                    current_sentence, current_labels = [], []

            # Add the last sentence if file does not end with an empty line
            if current_sentence:
                sentences.append(current_sentence)
                labels.append(current_labels)

        # self.logger.info(f"Read {len(sentences)} sentences from {file_path}")
        return sentences, labels, list(all_labels)

    def _preprocess_data(self) -> List[Dict]:
        """
        Preprocess the data by encoding the sentences and aligning the labels.
        """
        encoded_data = []

        for sentence_tokens, sentence_labels in zip(self.sentences, self.labels):
            encoding = self.tokenizer(
                sentence_tokens,
                is_split_into_words=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            labels = []
            word_ids = encoding.word_ids()
            previous_word_idx = None

            for word_idx in word_ids:
                if word_idx is None:
                    labels.append(-100)  # special tokens
                elif word_idx != previous_word_idx:
                    labels.append(self.label2id[sentence_labels[word_idx]])
                else:
                    labels.append(-100)  # subwords of a word
                previous_word_idx = word_idx

            item = {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': torch.tensor(labels)
            }
            encoded_data.append(item)

        return encoded_data

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        item = self.encoded_data[idx]
        input_ids = item['input_ids'].clone()  # Clone to avoid modifying original data

        # Apply [MASK] with probability `mask_out_prob`
        if self.mask_out_prob > 0:
            for i in range(len(input_ids)):
                if input_ids[i] != self.tokenizer.cls_token_id and \
                        input_ids[i] != self.tokenizer.sep_token_id and \
                        torch.rand(1).item() < self.mask_out_prob:
                    input_ids[i] = self.tokenizer.mask_token_id  # Replace with [MASK] token ID

        return {
            'input_ids': input_ids,
            'attention_mask': item['attention_mask'],
            'labels': item['labels']
        }

    def get_label_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of labels in the dataset.

        Returns:
            Dictionary mapping label names to their counts
        """
        label_counts = {}
        for item in self.encoded_data:
            labels = item['labels']
            for label in labels[labels != self.label_pad_token_id]:
                label_name = self.id2label[label.item()]
                label_counts[label_name] = label_counts.get(label_name, 0) + 1
        return label_counts

    def get_statistics(self) -> Dict:
        """
        Get various statistics about the dataset.

        Returns:
            Dictionary containing dataset statistics
        """
        seq_lengths = [len(s) for s in self.sentences]
        label_dist = self.get_label_distribution()

        return {
            'num_sequences': len(self.sentences),
            'avg_sequence_length': np.mean(seq_lengths),
            'max_sequence_length': max(seq_lengths),
            'num_labels': len(self.label2id),
            'label_distribution': label_dist,
            'total_tokens': sum(seq_lengths)
        }