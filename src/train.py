import time, random, os, csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from preprocess import prepare_data
from models import build_model

# Reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


def run_experiment(arch: str, seq_len: int = 50, clip: bool = False):
    """Train and evaluate one architecture with or without gradient clipping."""
    print(f"\n=== Training {arch.upper()} for sequence length {seq_len} (Gradient Clipping: {clip}) ===")

    # Step 1: Load data
    tokenizer, data = prepare_data(
        "C:/Users/dharm/sentiment_rnn_project/data/IMDB Dataset.csv",
        num_words=10000, seq_lengths=[seq_len])
    
    X, y = data[seq_len]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Step 2: Build model
    vocab_size = min(10000, len(tokenizer.word_index) + 1)
    model = build_model(
        arch=arch, vocab_size=vocab_size, seq_len=seq_len,
        emb_dim=100, hidden=64, layers=2, dropout=0.3, activation="tanh"
    )

    # Step 3: Compile model
    if clip:
        opt = tf.keras.optimizers.Adam(clipnorm=1.0)
        clip_flag = "Yes"
    else:
        opt = tf.keras.optimizers.Adam()
        clip_flag = "No"

    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    model.summary()

    # Step 4: Train
    start = time.time()
    hist = model.fit(Xtr, ytr, epochs=3, batch_size=32, validation_split=0.2, verbose=1)
    elapsed = time.time() - start

    # Step 5: Evaluate
    preds = (model.predict(Xte, verbose=0) > 0.5).astype("int32")
    acc = accuracy_score(yte, preds)
    f1 = f1_score(yte, preds, average="macro")

    print(f"{arch.upper()} ({clip_flag}) → Accuracy: {acc:.4f} | F1: {f1:.4f} | Time: {elapsed:.1f}s")

    # Step 6: Save data for later evaluation (only for one baseline model)
    RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

    if arch == "lstm" and seq_len == 50 and not clip:
        np.save(os.path.join(RESULTS_DIR, "lstm_history.npy"), hist.history)
        np.save(os.path.join(RESULTS_DIR, "yte.npy"), yte)
        np.save(os.path.join(RESULTS_DIR, "preds.npy"), preds)
        model.save(os.path.join(RESULTS_DIR, "best_lstm_model.h5"))
        print("\n Saved LSTM history, predictions, and model for evaluation.")

    return acc, f1, elapsed, clip_flag




# EXPERIMENT 1 — Architecture comparison (RNN, LSTM, BiLSTM) with/without clipping

def run_architecture_experiments():
    """Compare RNN, LSTM, and BiLSTM architectures (with and without gradient clipping)."""
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESULTS_DIR = os.path.join(ROOT_DIR, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    metrics_file = os.path.join(RESULTS_DIR, "metrics.csv")
    write_header = not os.path.exists(metrics_file)

    with open(metrics_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Model", "Activation", "Optimizer", "Seq Length",
                             "Grad Clipping", "Accuracy", "F1", "Epoch Time(s)"])

        for clip in [False, True]:
            for arch in ["rnn", "lstm", "bilstm"]:
                acc, f1, t, clip_flag = run_experiment(arch, seq_len=50, clip=clip)
                writer.writerow([arch.upper(), "tanh", "Adam", 50, clip_flag,
                                 f"{acc:.4f}", f"{f1:.4f}", f"{t:.1f}"])
                f.flush()
                os.fsync(f.fileno())

    print(f"\n Architecture experiment results saved to: {metrics_file}")


# EXPERIMENT 2 — Sequence length comparison (25, 50, 100) for LSTM
def run_seq_length_experiments():
    """Compare LSTM performance at different sequence lengths (25, 50, 100)."""
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESULTS_DIR = os.path.join(ROOT_DIR, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    metrics_file = os.path.join(RESULTS_DIR, "metrics_seq.csv")
    write_header = not os.path.exists(metrics_file)

    with open(metrics_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Model", "Activation", "Optimizer", "Seq Length",
                             "Grad Clipping", "Accuracy", "F1", "Epoch Time(s)"])

        for seq_len in [25, 50, 100]:
            for clip in [False, True]:
                acc, f1, t, clip_flag = run_experiment("lstm", seq_len=seq_len, clip=clip)
                writer.writerow(["LSTM", "tanh", "Adam", seq_len, clip_flag,
                                 f"{acc:.4f}", f"{f1:.4f}", f"{t:.1f}"])
                f.flush()
                os.fsync(f.fileno())

    print(f"\n Sequence length experiment results saved to: {metrics_file}")

def run_optimizer_activation_experiments():
    """Compare optimizers (Adam, RMSprop, SGD) and activations (tanh, relu, sigmoid) using LSTM."""
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESULTS_DIR = os.path.join(ROOT_DIR, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    metrics_file = os.path.join(RESULTS_DIR, "metrics_opt_act.csv")
    write_header = not os.path.exists(metrics_file)

    print("\n=== Starting Optimizer & Activation Experiment (LSTM, seq_len=50) ===")

    with open(metrics_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Model", "Activation", "Optimizer", "Seq Length",
                             "Grad Clipping", "Accuracy", "F1", "Epoch Time(s)"])

        for activation in ["tanh", "relu", "sigmoid"]:
            for optimizer_name in ["adam", "rmsprop", "sgd"]:
                print(f"\n--- Running LSTM with activation={activation}, optimizer={optimizer_name.upper()} ---")

                # Build optimizer dynamically
                if optimizer_name == "adam":
                    opt = tf.keras.optimizers.Adam()
                elif optimizer_name == "rmsprop":
                    opt = tf.keras.optimizers.RMSprop()
                elif optimizer_name == "sgd":
                    opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

                # Load data
                tokenizer, data = prepare_data(
                    "C:/Users/dharm/sentiment_rnn_project/data/IMDB Dataset.csv",
                    num_words=10000, seq_lengths=[50]
                )
                X, y = data[50]
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

                # Build model
                vocab_size = min(10000, len(tokenizer.word_index) + 1)
                model = build_model(
                    arch="lstm", vocab_size=vocab_size, seq_len=50,
                    emb_dim=100, hidden=64, layers=2, dropout=0.3, activation=activation
                )

                # Compile & train
                model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
                start = time.time()
                hist = model.fit(Xtr, ytr, epochs=3, batch_size=32, validation_split=0.2, verbose=1)
                elapsed = time.time() - start

                # Evaluate
                preds = (model.predict(Xte, verbose=0) > 0.5).astype("int32")
                acc = accuracy_score(yte, preds)
                f1 = f1_score(yte, preds, average="macro")

                writer.writerow(["LSTM", activation, optimizer_name.upper(), 50, "No",
                                 f"{acc:.4f}", f"{f1:.4f}", f"{elapsed:.1f}"])
                f.flush()
                os.fsync(f.fileno())

    print(f"\n Optimizer/Activation experiment completed. Results saved to: {metrics_file}")


# MAIN
def main():
    """Run both experiments in order."""
    print("\nStarting Experiment 1: Architecture Comparison")
    run_architecture_experiments()

    print("\nStarting Experiment 2: Sequence Length Comparison")
    run_seq_length_experiments()

    print("\nStarting Experiment 3: Optimizer and Activation Comparison")
    run_optimizer_activation_experiments()


if __name__ == "__main__":
    main()
