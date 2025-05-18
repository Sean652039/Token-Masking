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

        self.label2id = {"lang1": 0, "lang2": 1, "other": 2}
        self.id2label = {0: "lang1", 1: "lang2", 2: "other"}

        self.encoded_data = self._preprocess_data()

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

        return sentences, labels, list(all_labels)

    def _preprocess_data(self) -> List[Dict]:
        """
        Preprocess the data by encoding the sentences and aligning the labels.
        """
        encoded_data = []

        for sentence_tokens, sentence_labels in zip(self.sentences, self.labels):
            # 处理Qwen tokenizer的特殊情况
            encoding = self.tokenizer(
                sentence_tokens,
                is_split_into_words=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            labels = []

            # Get word_ids (index mapping token to original word)
            # Note: Some tokenizers may not have a word_ids method, so you need to implement the mapping logic yourself.
            try:
                word_ids = encoding.word_ids()
            except AttributeError:
                # If the tokenizer does not have a word_ids method, it needs to be mapped manually
                word_ids = self._map_tokens_to_words(encoding, sentence_tokens)

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

    def _map_tokens_to_words(self, encoding, sentence_tokens):
        """
        Manually mapping token to index of original word
        This is an alternative for when the tokenizer doesn't have a word_ids method
        """
        word_ids = []
        token_text = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])

        word_idx = 0
        current_word = ""

        for token in token_text:
            # Special token handling
            if token in [self.tokenizer.cls_token, self.tokenizer.sep_token,
                         self.tokenizer.pad_token, self.tokenizer.mask_token]:
                word_ids.append(None)
                continue

            # Remove ## prefixes (if any, e.g. subwords of BERT)
            if token.startswith("##"):
                token = token[2:]

            # For Qwen, different subword prefixes may need to be handled
            # Here it needs to be adapted to the situation

            current_word += token.replace("▁", "")  # Remove possible space markers (if any)

            if word_idx < len(sentence_tokens) and current_word in sentence_tokens[word_idx]:
                word_ids.append(word_idx)
                if current_word == sentence_tokens[word_idx]:
                    word_idx += 1
                    current_word = ""
            else:
                word_ids.append(None)  # If no words match

        return word_ids

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        item = self.encoded_data[idx]
        input_ids = item['input_ids'].clone()  # Clone to avoid modifying original data

        # Apply [MASK] with probability `mask_out_prob`
        if self.mask_out_prob > 0:
            # Make sure mask_token_id is a valid tensor or int that matches input_ids type
            mask_token_id = self.tokenizer.mask_token_id if hasattr(self.tokenizer, 'mask_token_id') else None

            # If no mask token is found, use the UNK token instead
            if mask_token_id is None:
                if hasattr(self.tokenizer, 'unk_token_id') and self.tokenizer.unk_token_id is not None:
                    mask_token_id = self.tokenizer.unk_token_id
                else:
                    # If no UNK token either, use a common token as fallback
                    mask_token_id = 0  # Often 0 is a safe choice for special tokens

            # Convert mask_token_id to a tensor with the same device as input_ids if needed
            if not isinstance(mask_token_id, torch.Tensor):
                mask_token_id = torch.tensor(mask_token_id, device=input_ids.device, dtype=input_ids.dtype)

            # Special tokens that shouldn't be masked
            special_tokens = set()
            for token_name in ['bos_token_id', 'eos_token_id', 'cls_token_id', 'sep_token_id', 'pad_token_id']:
                if hasattr(self.tokenizer, token_name):
                    token_id = getattr(self.tokenizer, token_name)
                    if token_id is not None:
                        special_tokens.add(token_id)

            for i in range(len(input_ids)):
                if input_ids[i].item() not in special_tokens and torch.rand(1).item() < self.mask_out_prob:
                    input_ids[i] = mask_token_id

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