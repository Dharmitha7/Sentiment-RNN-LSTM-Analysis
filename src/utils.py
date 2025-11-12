import pandas as pd
import os

def summarize_results(results_dir):
    """Summarize all experiment metrics into an easy-to-read report."""
    print("\n=== Experiment Summary ===")

    # 1. Load all result files
    files = {
        "Architecture": "metrics.csv",
        "Sequence Length": "metrics_seq.csv",
        "Optimizer-Activation": "metrics_opt_act.csv"
    }

    for name, fname in files.items():
        path = os.path.join(results_dir, fname)
        if not os.path.exists(path):
            print(f" Skipping {name} — {fname} not found.")
            continue

        df = pd.read_csv(path)
        print(f"\n {name} Results ({fname})")
        print(df.head())

        # 2. Compute summary stats
        best_row = df.loc[df["Accuracy"].idxmax()]
        print(f" Best Configuration → {best_row['Model']} | "
              f"Activation={best_row['Activation']}, Optimizer={best_row['Optimizer']}, "
              f"Seq={best_row['Seq Length']}, Clipping={best_row['Grad Clipping']}")
        print(f"   Accuracy={best_row['Accuracy']:.4f} | F1={best_row['F1']:.4f}\n")

        # 3. Save top configuration summary
        summary_file = os.path.join(results_dir, f"summary_{fname.replace('.csv', '.txt')}")
        with open(summary_file, "w") as f:
            f.write(f"=== {name} Summary ===\n")
            f.write(f"Best Configuration:\n")
            f.write(f"Model: {best_row['Model']}\n")
            f.write(f"Activation: {best_row['Activation']}\n")
            f.write(f"Optimizer: {best_row['Optimizer']}\n")
            f.write(f"Sequence Length: {best_row['Seq Length']}\n")
            f.write(f"Gradient Clipping: {best_row['Grad Clipping']}\n")
            f.write(f"Accuracy: {best_row['Accuracy']:.4f}\n")
            f.write(f"F1 Score: {best_row['F1']:.4f}\n")
            f.write(f"Epoch Time: {best_row['Epoch Time(s)']:.1f} s\n")
        print(f" Summary saved to: {summary_file}")

    print("\n All summaries generated.")


if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESULTS_DIR = os.path.join(ROOT_DIR, "results")
    summarize_results(RESULTS_DIR)
