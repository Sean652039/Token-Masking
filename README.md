# Demo of Code-Switching Language Identification Mask Out

# [Outputs](https://drive.google.com/file/d/1qPCmsRjCW8hDfxfMY-CG04E7s-oXGCqL/view?usp=sharing)

This is the demo and also proof of the concept

## File Structure

```
project_root/
├── CS_Dataset.py
├── train.py
├── loss.py
├── utils/
│   └── build_datasets.py
├── lid_spaeng
│   ├── dev.conll
│   ├── test.conll
│   └── train.conll
├── lid_nepeng
│   ├── dev.conll
│   ├── test.conll
│   └── train.conll
├── logs
├── outputs
└── plot
```

`analysis.py` is for analyzing the information about the dataset, like number of classes and label distribution.

We are using [LinCE datasets](https://ritual.uh.edu/lince/datasets), which are `lid_nepeng` and  `lid_spaeng` with train, dev and test datasets in .conll format.

## Train the model

```bash
python train.py
```

Train the model on the English-Spanish with masking out and test on the English-Hindi.

## Results
![Loss](plots/training_curves.png)

- Mask Out Prob = 0

```txt
2024-11-16 14:05:43,374 [INFO] Evaluating on test set...
2024-11-16 14:05:44,601 [INFO] Test Results: F1 Macro = 0.8221, F1 Weighted = 0.8149
```

- Mask Out Prob = 0.15

```txt
2024-11-16 13:02:22,601 [INFO] Evaluating on test set...
2024-11-16 13:02:23,879 [INFO] Test Results: F1 Macro = 0.8263, F1 Weighted = 0.8203
```

## Issue

### 1. Only calculated the weights depend on the training data.
![label_distribution](plots/label_distribution.png)

### 



