# Classifying IMDb Movie Reviews with BERT Transformers

**Status:** Work in progress · **Python / PyTorch / Hugging Face**

A notebook-based natural language processing project exploring **multiclass classification of IMDb movie reviews by the aspect being discussed**. The task is to classify review text into one of three categories: **Cinematography**, **Characters**, or **Story**.

The project begins with an embedding-based neural-network baseline and is intended to progress to fine-tuning a **TinyBERT** transformer and comparing model performance.

> **Current progress:** Data exploration and preprocessing, vocabulary construction, and the baseline PyTorch model definition/optimiser setup are implemented. Baseline training/evaluation and TinyBERT fine-tuning/evaluation sections remain to be completed. The notebook includes instructional prompts and example results from the original course; these should **not** be interpreted as results obtained from the code in this repository.

## Project goals

- Explore a labelled movie-review dataset and its aspect classes.
- Prepare text for a conventional neural-network classifier using tokenisation, encoding, and fixed-length sequences.
- Train and evaluate a simple embedding-based baseline.
- Fine-tune TinyBERT for the same multiclass task.
- Compare models using a held-out test set, including accuracy, precision, recall, F1-score and a confusion matrix.

## Dataset

Source: [Lowerated/lm6-movies-reviews-aspects](https://huggingface.co/datasets/Lowerated/lm6-movies-reviews-aspects) on Hugging Face.

The notebook loads the dataset with:

```python
from datasets import load_dataset
ds = load_dataset("Lowerated/lm6-movies-reviews-aspects")
```

The working data include a review text (`review`), a categorical aspect label (`aspect`), and an encoded label (`aspect_encoded`). The notebook uses the dataset's training split. Downloaded data are **not committed** to this repository. The notebook's instructional text also refers to local CSV files, but the implemented loader uses Hugging Face instead.

## Approach

### 1. Inspect and preprocess text — implemented

- Load and inspect the training records and aspect frequencies.
- Lowercase and extract word tokens with a regular expression.
- Build a vocabulary from the 1,000 most common words, plus `<unk>` and `<pad>` tokens.
- Convert words to integer IDs; truncate or pad each review to 128 tokens.
- Construct PyTorch tensors and a training `DataLoader` (batch size 16).

### 2. Embedding-based neural network — defined; training pending

The baseline model is a compact PyTorch classifier:

```text
Token IDs
   ↓
Embedding(vocabulary_size, 50)
   ↓
Mean over token positions
   ↓
Linear(50, 100) → ReLU
   ↓
Linear(100, 3)
   ↓
Class logits
```

The notebook defines cross-entropy loss and Adam with a learning rate of `0.005`.

**Limitation:** The baseline currently averages *all* token embeddings, including padding positions; padding-aware mean pooling would provide a cleaner representation. It also does not explicitly model word order.

### 3. TinyBERT transformer — planned, not yet implemented

The remaining course exercises describe loading and fine-tuning a pretrained TinyBERT classifier with Hugging Face Transformers, preparing attention masks, and evaluating the model against the baseline. Those code cells are currently empty.

## Repository structure

```text
Classifying_IMDB_Movie_Reviews_with_BERT_Transformers/
├── imdb_aspect_classification.ipynb  # Main working notebook
├── README.md                         # Project overview and reproducibility notes
├── requirements.txt                  # Python dependencies
└── .gitignore                        # Excludes caches, datasets and model weights
```

## Getting started

You will need Python and Jupyter. PyTorch and transformer installation may depend on your hardware/CUDA setup; see the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for GPU-specific instructions.

```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/Classifying_IMDB_Movie_Reviews_with_BERT_Transformers.git
cd Classifying_IMDB_Movie_Reviews_with_BERT_Transformers
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab
```

Open `imdb_aspect_classification.ipynb` and run cells in order. Internet access is needed the first time the Hugging Face dataset is fetched. Running the full training split may require substantial RAM; the notebook has not been end-to-end validated in this repository.

> The clone URL is a template **until you create and push the repository on GitHub**. Replace `<YOUR_GITHUB_USERNAME>` with your account name.

## Evaluation plan

After completing the training and test-processing cells, report performance for **both** models on the *same* untouched test split:

| Model | Accuracy | Macro F1 | Status |
|---|---:|---:|---|
| Mean-pooled embedding network | — | — | Not yet evaluated |
| Fine-tuned TinyBERT | — | — | Not yet implemented |

Do not interpret the sample accuracy figures embedded in course instructions as independently reproduced experimental results.

## Next improvements

1. **Complete the baseline:** implement training, test preprocessing, and test-set inference.
2. **Implement TinyBERT:** select and document a specific checkpoint, prepare attention masks, train, and evaluate.
3. **Improve pooling:** exclude `<pad>` tokens from the baseline's mean embedding calculation.
4. **Use a validation split:** tune model settings on validation data while preserving the test set for final reporting.
5. **Strengthen evaluation:** include macro/weighted F1, per-class errors, confusion matrices, and a side-by-side results table.
6. **Improve reproducibility:** pin tested package versions, record model and data versions, and save training configurations/seeds.
7. **Go beyond the tutorial:** run an ablation on vocabulary size, sequence length, and padding-aware pooling; compare against a TF–IDF + logistic-regression baseline.

## Acknowledgements and attribution

This project is based on the Codecademy learning exercise [*Classifying IMDB Movie Reviews with BERT Transformers*](https://www.codecademy.com/content-items/29838c7636654e48ac72458af6373d5d). The notebook retains course-authored instructional material and links. The movie-review dataset is sourced from [Lowerated on Hugging Face](https://huggingface.co/datasets/Lowerated/lm6-movies-reviews-aspects).

No independent licence for the course material or dataset is asserted here; check their terms before redistributing modified materials or the raw dataset.
