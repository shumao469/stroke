import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc


# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
def _set_plot_style():
    """Robust plotting style for different Matplotlib versions."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        # Fallback for older Matplotlib
        try:
            plt.style.use("seaborn-whitegrid")
        except Exception:
            pass
    sns.set_context("talk", font_scale=1.15)  # optimized for presentation / video


_set_plot_style()

COLORS = {
    "HS": "#e74c3c",
    "ZS": "#3498db",
    "NC": "#2c3e50",
    "SAFE": "#27ae60",
    "RISK": "#c0392b",
}


# -----------------------------------------------------------------------------
# DATA SIMULATION ENGINE
# -----------------------------------------------------------------------------
class DataEngine:
    """Generates synthetic metabolomics data based on simplified biological logic.

    NOTE: This is a demonstration generator only (not real patient data).
    """

    @staticmethod
    def generate_triage_data(n=500, seed=42):
        """Generate HS vs ZS vs NC data for ED triage (binary target: HS vs non-HS)."""
        rng = np.random.default_rng(seed)

        rows = []
        groups = ["NC", "HS", "ZS"]
        per = max(1, n // 3)

        for g in groups:
            for _ in range(per):
                row = {"Group": g}
                # HS high markers (inflammation/bleeding signature)
                if g == "HS":
                    row["RvD5"] = rng.normal(8.5, 1.2)
                    row["8_MKNA"] = rng.normal(7.8, 1.0)
                    row["N_AcCad"] = rng.normal(9.2, 1.5)
                elif g == "ZS":
                    row["RvD5"] = rng.normal(5.0, 1.2)
                    row["8_MKNA"] = rng.normal(4.5, 1.0)
                    row["N_AcCad"] = rng.normal(5.5, 1.2)
                else:  # NC
                    row["RvD5"] = rng.normal(3.0, 0.8)
                    row["8_MKNA"] = rng.normal(2.5, 0.8)
                    row["N_AcCad"] = rng.normal(3.2, 0.8)
                rows.append(row)

        df = pd.DataFrame(rows)
        df["Target"] = (df["Group"] == "HS").astype(int)
        return df

    @staticmethod
    def generate_prognosis_data(n=500, seed=123):
        """Generate hematoma expansion and outcome data."""
        rng = np.random.default_rng(seed)

        rvd5 = rng.normal(6.0, 2.0, n)
        n_accad = rng.normal(5.5, 1.8, n)
        mkna_8 = rng.normal(5.0, 1.5, n)

        # Expansion logic
        logit_exp = -3.5 + 0.5 * rvd5 + 0.4 * n_accad + rng.normal(0, 0.5, n)
        prob_exp = 1 / (1 + np.exp(-logit_exp))
        expansion = (rng.random(n) < prob_exp).astype(int)

        # Outcome logic (mRS poor outcome)
        logit_mrs = -4.0 + 0.6 * mkna_8 + 1.5 * expansion + 0.3 * rvd5 + rng.normal(0, 0.5, n)
        prob_mrs = 1 / (1 + np.exp(-logit_mrs))
        poor_outcome = (rng.random(n) < prob_mrs).astype(int)

        return pd.DataFrame(
            {
                "RvD5": rvd5,
                "N_AcCad": n_accad,
                "8_MKNA": mkna_8,
                "Hematoma_Expansion": expansion,
                "Poor_Outcome_mRS": poor_outcome,
            }
        )


# -----------------------------------------------------------------------------
# VISUALIZATION MODULES
# -----------------------------------------------------------------------------
class Visualizer:
    @staticmethod
    def plot_roc(y_true, y_prob):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color=COLORS["HS"], lw=3, label=f"Ocular Met-Score (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
        plt.title("ED Triage Performance: HS Detection", fontweight="bold")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_dca(y_true, y_prob):
        """Decision curve analysis (net benefit) for a binary classifier."""
        thresholds = np.linspace(0.01, 0.99, 100)
        n = len(y_true)
        net_benefits = []

        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob)

        for t in thresholds:
            pred_pos = y_prob >= t
            tp = np.sum(pred_pos & (y_true == 1))
            fp = np.sum(pred_pos & (y_true == 0))
            nb = (tp / n) - (fp / n) * (t / (1 - t))
            net_benefits.append(nb)

        prevalence = np.mean(y_true)
        nb_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, net_benefits, color=COLORS["HS"], lw=3, label="Ocular Model")
        plt.plot(thresholds, nb_all, lw=1, linestyle="--", label="Treat All")
        plt.axhline(y=0, lw=1, linestyle="-", label="Treat None")

        plt.xlim(0, 0.8)
        plt.ylim(-0.05, 0.5)
        plt.title("Decision Curve Analysis (Clinical Net Benefit)", fontweight="bold")
        plt.xlabel("Threshold Probability")
        plt.ylabel("Net Benefit")
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_calibration_strata(y_true, y_prob):
        """Calibration curve + simple risk stratification plot."""
        plt.figure(figsize=(14, 6))

        # Calibration
        ax1 = plt.subplot(1, 2, 1)
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        ax1.plot(prob_pred, prob_true, marker="o", lw=2, color=COLORS["HS"], label="Model")
        ax1.plot([0, 1], [0, 1], "--", label="Ideal")
        ax1.set_title("Calibration Curve (Reliability)", fontweight="bold")
        ax1.set_xlabel("Predicted Probability")
        ax1.set_ylabel("Observed Rate")
        ax1.legend()

        # Risk stratification
        ax2 = plt.subplot(1, 2, 2)
        risk_bins = pd.cut(y_prob, bins=[0, 0.3, 0.7, 1.0], labels=["Low", "Moderate", "High"])
        strat_df = pd.DataFrame({"Risk": risk_bins, "Event": y_true})
        means = strat_df.groupby("Risk", observed=False)["Event"].mean() * 100

        colors = [COLORS["SAFE"], "#f39c12", COLORS["RISK"]]
        bars = ax2.bar(means.index, means.values, color=colors, width=0.6)

        ax2.set_title("Risk Stratification", fontweight="bold")
        ax2.set_ylabel("Expansion Rate (%)")
        ax2.set_ylim(0, 100)

        for bar in bars:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f"{bar.get_height():.1f}%",
                ha="center",
                fontweight="bold",
            )

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_waterfall(feature_names, contributions, base_value, patient_id):
        """SHAP-style waterfall plot (demo, no SHAP dependency)."""
        plt.figure(figsize=(10, 6))

        contributions = np.asarray(contributions, dtype=float)
        indices = np.argsort(np.abs(contributions))
        names = [feature_names[i] for i in indices]
        vals = [contributions[i] for i in indices]

        running_sum = float(base_value)
        for i, (n, v) in enumerate(zip(names, vals)):
            c = "#ff0051" if v > 0 else "#008bfb"
            plt.barh(i, v, left=running_sum, color=c, height=0.6)
            plt.text(running_sum + v, i, f"{v:+.2f}", va="center", fontsize=10, fontweight="bold", color=c)
            if i < len(names) - 1:
                plt.plot([running_sum + v, running_sum + v], [i, i + 1], "gray", lw=0.5)
            running_sum += v

        plt.yticks(range(len(names)), names)
        plt.axvline(x=base_value, color="gray", linestyle="--")
        plt.xlabel("Log-Odds Contribution")
        plt.title(f"SHAP-style Explanation: Patient {patient_id}", fontweight="bold")
        plt.tight_layout()
        plt.show()


# -----------------------------------------------------------------------------
# MAIN PIPELINE CONTROLLER
# -----------------------------------------------------------------------------
def run_triage_mode():
    print("\n>>> [MODE 1] INITIALIZING ED TRIAGE PIPELINE...")
    df = DataEngine.generate_triage_data()

    X = df[["RvD5", "8_MKNA", "N_AcCad"]]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    print("    Step 1: Analyzing ROC Performance...")
    Visualizer.plot_roc(y_test, y_prob)

    print("    Step 2: Calculating Clinical Net Benefit (DCA)...")
    Visualizer.plot_dca(y_test, y_prob)

    print(">>> [COMPLETED] Triage module finished.\n")


def run_expansion_mode():
    print("\n>>> [MODE 2] INITIALIZING HEMATOMA EXPANSION RISK PIPELINE...")
    df = DataEngine.generate_prognosis_data()

    X = df[["RvD5", "N_AcCad"]]
    y = df["Hematoma_Expansion"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    print("    Step 1: Generating Calibration and Risk Strata Plots...")
    Visualizer.plot_calibration_strata(y_test, y_prob)

    print(">>> [COMPLETED] Risk Assessment finished.\n")


def run_outcome_mode():
    print("\n>>> [MODE 3] INITIALIZING OUTCOME PREDICTION (mRS)...")
    df = DataEngine.generate_prognosis_data()

    X = df[["RvD5", "8_MKNA", "N_AcCad"]]
    y = df["Poor_Outcome_mRS"]

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    # Simulate single patient explanation
    pat = X.iloc[0]
    base_val = -1.5
    weights = [0.4, 0.6, 0.2]
    contribs = [(pat[col] - X[col].mean()) * w for col, w in zip(X.columns, weights)]
    names = [f"{col}={pat[col]:.1f}" for col in X.columns]

    patient_id = f"PT-{np.random.randint(1000, 9999)}"
    print(f"    Step 1: Explaining Prediction for Patient {patient_id}...")
    Visualizer.plot_waterfall(names, contribs, base_val, patient_id)

    print(">>> [COMPLETED] Prognosis module finished.\n")


def main():
    print("================================================================")
    print("   OCULAR METABOLOMICS STROKE PIPELINE (CLINICAL DEMO v1.0)     ")
    print("================================================================")

    while True:
        print("\nSelect Operation Mode:")
        print(" [1] ED Triage (ROC + DCA)")
        print(" [2] Hematoma Expansion Risk (Calibration + Strata)")
        print(" [3] Outcome Prediction (SHAP-style Waterfall)")
        print(" [Q] Quit")

        choice = input("\nEnter selection > ").strip().upper()

        if choice == "1":
            run_triage_mode()
        elif choice == "2":
            run_expansion_mode()
        elif choice == "3":
            run_outcome_mode()
        elif choice == "Q":
            print("Exiting...")
            break
        else:
            print("Invalid selection.")


if __name__ == "__main__":
    main()
