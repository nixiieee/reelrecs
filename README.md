# 🎬 ReelRecs: Movie Recommendation System

**ReelRecs** is a neural network-based movie recommendation system built for an online cinema platform. The system leverages a Deep Structured Semantic Model (DSSM) to learn user-item interactions and generate personalized film recommendations.

---

## 🚀 Project Structure

```
.
├── .gitignore               # General ignore rules
├── data/.gitignore          # Placeholder for data files
├── model_weights/.gitignore # Placeholder for model weights
├── EDA.ipynb                # Exploratory data analysis
├── preprocessing.ipynb      # Data preprocessing steps
├── train_dssm.ipynb         # DSSM model training pipeline
├── early_stopping.py        # Custom early stopping callback
├── study_results.json       # Results of hyperparameter optimization
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
```

---

## 🧠 Approach

This project implements a **DSSM-based architecture** to embed users and movies into a shared latent space, enabling efficient retrieval of relevant content.

The training pipeline involves:

* Data preprocessing & filtering
* Feature engineering (user/movie embeddings)
* Training with early stopping
* Logging & evaluation

Hyperparameter tuning and experiment tracking were conducted using [Weights & Biases (wandb)](https://wandb.ai/).

> ⚠️ The dataset and model weights are **not included** in this repository and will not be published.

---

## 📦 Setup

Ensure you're using **Python 3.10**.

Create and activate a virtual environment:

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or using [`uv`](https://github.com/astral-sh/uv):

```bash
uv venv -p 3.10
source .venv/bin/activate
uv install -r requirements.txt
```

---

## 📁 Data

Place your data files inside the `data/` directory.
Example (for the KION dataset):

```
data/
└── data_kion/
    ├── interactions.csv
    └── metadata.csv
```

> Ensure the data format matches the expectations defined in `preprocessing.ipynb`.

---

## 📊 Notebooks

* **EDA.ipynb** — initial data analysis and visualizations
* **preprocessing.ipynb** — cleaning and transforming the dataset
* **train\_dssm.ipynb** — model training and evaluation

---

## 📊 Logging & Tracking with Weights & Biases (W&B)

During training, the model logs metrics and visualizations to [Weights & Biases](https://wandb.ai/):

- Training & validation loss
- Accuracy per epoch
- Confusion matrix & misclassification analysis

To enable logging, make sure you are logged in:

```bash
wandb login
```

By default, W&B is integrated in the training script:

```python
import wandb
wandb.init(project="whisper-emotion")
```

You can monitor training in real time at [wandb.ai](https://wandb.ai/) or in your terminal.

---

## 📫 Contact

For questions or collaboration ideas, feel free to reach out via GitHub Issues.

---

> Made with ❤️ for movie lovers and ML enthusiasts
