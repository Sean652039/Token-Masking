# Token Masking Improves Transformer-Based Text Classification

This repository provides the implementation of **Token Masking Regularization**, a simple yet effective training strategy to improve Transformer-based text classifiers by randomly masking input tokens during training.

## Overview

Transformer models like BERT have achieved state-of-the-art performance on various text classification tasks. In this work, we explore a theoretically motivated **token masking regularization** technique that introduces stochastic input perturbations by randomly replacing input tokens with a `[MASK]` token at probability `p`.

## Experiments

We evaluate the proposed method on:

- **Language Identification (LID)** (including cross-lingual transfer)
- **Sentiment Analysis (SA)**

Models tested:
- [mBERT](https://huggingface.co/bert-base-multilingual-cased)
- [Qwen2.5-0.5B](https://huggingface.co/Qwen)
- [TinyLlama-1.1B](https://huggingface.co/TinyLlama)

Results show consistent gains over standard regularization techniques (e.g., dropout), with `p = 0.1` as a strong default.

#### [Outputs](https://drive.google.com/file/d/1qPCmsRjCW8hDfxfMY-CG04E7s-oXGCqL/view?usp=sharing) (Only the language identification outputs are provided due to Google Cloud storage limitations. Contact me if you need additional data.)

## Usage

### Installation

```
pip install -r requirements.txt
```

### Navigate to Task Folder

Enter the folder corresponding to the specific task:

```bash
cd <task_folder>
# e.g., cd language_identification
```

### Training

Run the training script:

```bash
python train.py
```

## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{xu2025tokenmaskingimprovestransformerbased,
      title={Token Masking Improves Transformer-Based Text Classification}, 
      author={Xianglong Xu and John Bowen and Rojin Taheri},
      year={2025},
      eprint={2505.11746},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.11746}, 
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
