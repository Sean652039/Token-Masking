from torch.utils.data import Dataset
import torch
import numpy as np
from typing import List, Dict
import logging
import os
import re


class SADataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=40, mask_out_prob=0.15):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_out_prob = mask_out_prob

        self.sentences, self.labels, all_labels = self._read_conll_file(file_path)

        label2id = {label: idx for idx, label in enumerate(sorted(set(all_labels)))}
        # # sort by label name
        self.label2id = dict(sorted(label2id.items()))
        self.id2label = {idx: label for label, idx in self.label2id.items()}

    def _read_conll_file(self, file_path):
        sentences = []
        labels = []
        all_labels = set()
        current_sentence = []
        current_label = None

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                # get the sentence label
                if line.startswith("# sent_enum"):
                    # sent_enum = 1 positive"
                    parts = line.split()
                    if parts:
                        # if we have a current sentence, save it before starting a new one
                        if current_sentence and current_label is not None:
                            sentences.append(' '.join(current_sentence))
                            labels.append(current_label)
                            all_labels.add(current_label)

                        # new sentence
                        current_label = parts[-1]  # e.g., "positive"
                        current_sentence = []

                # get the sentence tokens
                elif line and not line.startswith("#"):
                    #"After lang1"
                    parts = line.split()
                    if len(parts) >= 2:
                        token = parts[0]
                        # lang_tag = parts[1]  # Language labels, if they need to be used
                        current_sentence.append(token)

            # end of file, save the last sentence if it exists
            if current_sentence and current_label is not None:
                sentences.append(' '.join(current_sentence))
                labels.append(current_label)

        return sentences, labels, list(all_labels)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]

        # Tokenize the sentence
        encoding = self.tokenizer(
            sentence,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            return_attention_mask=True
        )

        # Apply mask out
        input_ids = encoding['input_ids'].squeeze(0).numpy()
        # NO masking for special tokens - Qwen may use different special token IDs
        mask = np.random.rand(len(input_ids)) < self.mask_out_prob

        for i in range(len(input_ids)):
            if (input_ids[i] not in [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id] and mask[i]):
                # Get mask token ID with proper fallbacks
                mask_token_id = None
                if hasattr(self.tokenizer, 'mask_token_id') and self.tokenizer.mask_token_id is not None:
                    mask_token_id = self.tokenizer.mask_token_id
                elif hasattr(self.tokenizer, 'unk_token_id') and self.tokenizer.unk_token_id is not None:
                    mask_token_id = self.tokenizer.unk_token_id
                else:
                    mask_token_id = 0  # Fallback to a safe token ID

                input_ids[i] = mask_token_id
        # Re-assign the masked input_ids back to the encoding
        encoding['input_ids'] = torch.tensor(input_ids).unsqueeze(0)

        label_id = self.label2id[label]

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_id, dtype=torch.long)
        }

    def get_label_distribution(self) -> Dict[str, int]:
        """
        Returns the distribution of labels in the dataset.
        """
        label_distribution = {label: 0 for label in self.label2id.keys()}
        for label in self.labels:
            label_distribution[label] += 1
        return label_distribution

    def get_statistics(self) -> Dict:
        """
        Get various statistics about the dataset.

        Returns:
            Dictionary containing dataset statistics
        """
        seq_lengths = [len(s.split()) for s in self.sentences]
        label_dist = self.get_label_distribution()

        return {
            'num_sequences': len(self.sentences),
            'avg_sequence_length': np.mean(seq_lengths),
            'max_sequence_length': max(seq_lengths),
            'num_labels': len(self.label2id),
            'label_distribution': label_dist,
            'total_tokens': sum(seq_lengths)
        }