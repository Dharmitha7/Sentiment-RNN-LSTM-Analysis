# Sentiment Classification with RNN, LSTM, and BiLSTM  
Author: *Dharmitha Niteesha Vattikonda*  

---

##  Overview
This project explores **recurrent neural network architectures** (RNN, LSTM, and Bidirectional LSTM) for **sentiment classification** on the IMDB movie reviews dataset.  
It evaluates multiple configurations across:

- **Architecture:** RNN, LSTM, BiLSTM  
- **Activation functions:** `tanh`, `relu`, `sigmoid`  
- **Optimizers:** `Adam`, `RMSProp`, `SGD`  
- **Sequence lengths:** 25, 50, 100  
- **Stability strategy:** With and without gradient clipping  

The goal is to compare how these choices affect accuracy, F1-score, training time, and overall stability.

---

##  Setup Instructions

###  Python Environment
This project was tested with:

```

Python 3.10+
TensorFlow 2.12+
scikit-learn 1.5+
matplotlib 3.9+
seaborn 0.13+
pandas 2.2+
numpy 1.26+

````

You can install dependencies manually or using `pip`:

```bash
pip install -r requirements.txt
````

---

##  Project Structure

```
sentiment_rnn_project/
│
├── data/
│   └── IMDB Dataset.csv
│
├── results/
│   ├── metrics.csv
│   ├── metrics_seq.csv
│   ├── metrics_opt_act.csv
│   ├── plots/
│   │   ├── architecture_accuracy.png
│   │   ├── seq_accuracy.png
│   │   ├── opt_activation_f1.png
│   │   └── ...
│   └── summaries/
│
├── src/
│   ├── preprocess.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
└── README.md
```

---

##  Running the Project

### 1.Train all models and save results

Run from inside the `src/` directory:

```bash
python train.py
```

This will:

* Train and evaluate all RNN, LSTM, BiLSTM models
* Run experiments for sequence length and optimizers
* Save:

  * Trained model (`best_lstm_model.h5`)
  * Predictions (`preds.npy`)
  * Evaluation data (`yte.npy`, `lstm_history.npy`)
  * CSV logs in `/results/`

---

### 2. Generate Evaluation Plots

After training completes, run:

```bash
python evaluate.py
```

This script:

* Reads your result CSVs
* Generates bar, line, scatter, and heatmap plots
* Saves all images inside `/results/plots/`

---

### 3. Summarize Best Configurations (optional)

To print and save best-performing configurations:

```bash
python utils.py
```

This produces text summaries (e.g. `summary_metrics.txt`) showing top configurations for each experiment.

---

## Expected Runtime

| Model                 | Dataset Size | Avg Epoch Time | Total Runtime           |
| --------------------- | ------------ | -------------- | ----------------------- |
| RNN                   | 50 seq       | ~40 sec/epoch  | ~4 min                  |
| LSTM                  | 50 seq       | ~80 sec/epoch  | ~8–10 min              |
| BiLSTM                | 50 seq       | ~130 sec/epoch  | ~13-15 min                 |
| Full experiment suite | —            | —              | ~30–40 min total on CPU |

>  Using a GPU (e.g., on Google Colab or Kaggle) can reduce runtime by 3–4×.

---

##  Output Files

| File                                    | Description                                          |
| --------------------------------------- | ---------------------------------------------------- |
| `results/metrics.csv`                   | Accuracy/F1/time for RNN, LSTM, BiLSTM               |
| `results/metrics_seq.csv`               | Performance across sequence lengths                  |
| `results/metrics_opt_act.csv`           | Comparison of optimizers & activations               |
| `results/plots/`                        | Visualizations (accuracy, F1, learning curves, etc.) |
| `results/best_lstm_model.h5`            | Saved trained model                                  |
| `results/lstm_history.npy`              | Learning curve data                                  |
| `results/preds.npy` / `results/yte.npy` | Predictions and true labels                          |
| `results/summary_*.txt`                 | Text summaries of best configurations                |

---

##  Key Findings (Summary)

* **Best model:** LSTM (tanh, Adam, sequence length = 100)
* **Best F1-score:** ≈ 0.83
* **BiLSTM** adds training time but limited accuracy gain.
* **Adam and RMSProp** outperform SGD for stability and convergence.
* **Gradient clipping** slightly improves consistency but not accuracy.


 