import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix

# Style setup
sns.set(style="whitegrid", font_scale=1.2)

# Directory setup
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ==== 1. Architecture Comparison ====
df_arch = pd.read_csv(os.path.join(RESULTS_DIR, "metrics.csv"))
plt.figure(figsize=(8, 5))
sns.barplot(data=df_arch, x="Model", y="Accuracy", hue="Grad Clipping", palette="pastel")
plt.title("Architecture Comparison (Accuracy)")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "architecture_accuracy.png"))
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(data=df_arch, x="Model", y="F1", hue="Grad Clipping", palette="muted")
plt.title("Architecture Comparison (F1-Score)")
plt.ylabel("F1-Score")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "architecture_f1.png"))
plt.show()

# ==== 2. Sequence Length Comparison ====
df_seq = pd.read_csv(os.path.join(RESULTS_DIR, "metrics_seq.csv"))
plt.figure(figsize=(8, 5))
sns.lineplot(data=df_seq, x="Seq Length", y="Accuracy", hue="Grad Clipping", marker="o")
plt.title("LSTM Accuracy by Sequence Length")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "seq_accuracy.png"))
plt.show()

plt.figure(figsize=(8, 5))
sns.lineplot(data=df_seq, x="Seq Length", y="F1", hue="Grad Clipping", marker="o")
plt.title("LSTM F1-Score by Sequence Length")
plt.ylabel("F1-Score")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "seq_f1.png"))
plt.show()

# ==== 3. Optimizer & Activation Comparison ====
df_opt = pd.read_csv(os.path.join(RESULTS_DIR, "metrics_opt_act.csv"))
plt.figure(figsize=(10, 6))
sns.barplot(data=df_opt, x="Optimizer", y="Accuracy", hue="Activation", palette="coolwarm")
plt.title("Optimizer & Activation Comparison (Accuracy)")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "opt_activation_accuracy.png"))
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=df_opt, x="Optimizer", y="F1", hue="Activation", palette="coolwarm")
plt.title("Optimizer & Activation Comparison (F1-Score)")
plt.ylabel("F1-Score")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "opt_activation_f1.png"))
plt.show()

print("\n✅ Plots for architecture, sequence length, and optimizer-activation comparisons saved in:", PLOTS_DIR)

# ==== 4. Learning Curves ====
# Load saved data
history = np.load(os.path.join(RESULTS_DIR, "lstm_history.npy"), allow_pickle=True).item()
yte = np.load(os.path.join(RESULTS_DIR, "yte.npy"))
preds = np.load(os.path.join(RESULTS_DIR, "preds.npy"))

plt.figure(figsize=(8, 5))
plt.plot(history["accuracy"], label="Train Accuracy")
plt.plot(history["val_accuracy"], label="Validation Accuracy")
plt.title("Learning Curve - LSTM (seq_len=50, no clipping)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "lstm_learning_curve.png"))
plt.show()

# ==== 5. Confusion Matrix for Best Model ====
cm = confusion_matrix(yte, preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.title("Confusion Matrix - Best Model (LSTM, tanh, Adam, seq_len=50)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix_best.png"))
plt.show()

# ==== 6. Training Time vs Accuracy Trade-off ====
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_arch, x="Epoch Time(s)", y="Accuracy", hue="Model",
                style="Grad Clipping", s=100)
plt.title("Training Time vs Accuracy Trade-off")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "time_vs_accuracy.png"))
plt.show()

# ==== 7. Performance Heatmap ====
heat_df = df_opt.pivot(index="Activation", columns="Optimizer", values="Accuracy")
sns.heatmap(heat_df, annot=True, cmap="YlGnBu")
plt.title("Optimizer vs Activation (Accuracy Heatmap)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "opt_act_heatmap.png"))
plt.show()

print("\n All plots successfully generated and saved to:", PLOTS_DIR)
