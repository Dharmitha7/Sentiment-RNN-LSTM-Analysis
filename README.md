# Sentiment Classification Tradeoffs with RNN, LSTM, and BiLSTM 
Author: *Dharmitha Niteesha Vattikonda*  

---

##  Overview
This project investigates how different recurrent neural network architectures perform on a real-world sentiment classification task using the IMDB movie reviews dataset. The goal is not only to compare RNN, LSTM, and Bidirectional LSTM models, but to understand how architectural choices, sequence length, optimization strategy, and gradient stability affect performance, training time, and reliability.

Through a controlled experimental setup, the project evaluates tradeoffs between model complexity, convergence behavior, and predictive quality, with an emphasis on selecting configurations that balance accuracy, stability, and computational cost.

---

## Problem Statement

Sentiment classification on long-form text presents challenges related to sequence modeling, gradient stability, and computational efficiency. While simple RNNs struggle with long-range dependencies, more complex architectures such as LSTMs and Bidirectional LSTMs introduce additional training cost and complexity.

Choosing an appropriate architecture and training configuration requires understanding the tradeoffs between accuracy, stability, and runtime. This project addresses that problem by systematically evaluating recurrent model variants under different training conditions to identify practical, high-performing configurations.

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

##  Insights & Tradeoffs

* LSTM with tanh activation and Adam optimizer provides the best balance of accuracy and training stability.
* Bidirectional LSTMs increase training time significantly while offering marginal accuracy improvements on this dataset.
* Adam and RMSProp converge faster and more reliably than SGD, especially for longer sequences.
* Gradient clipping improves training stability but does not materially improve final accuracy.

---

## Production Considerations

Based on these results, a standard LSTM with moderate sequence length would be preferred in a production setting due to its balance of performance and computational cost. More complex architectures such as BiLSTMs may be justified only when marginal accuracy gains outweigh increased latency and resource usage.

Further improvements could include pretrained embeddings, transformer-based architectures, or model distillation for deployment efficiency.



 
