"""Stage 04 — reproduce Table 8, Table 9 (validation), a multi-seed robustness
study, and write the Kaggle submission files.

Outputs:
  reports/tables/table8.{csv,md}        all 5 variants x both tasks (seed 42)
  reports/tables/robustness.{csv,md}    M2/M4 over seeds 42..51 (mean +/- std)
  reports/tables/table9.{csv,md}        selected models + OLS positioning baseline
  reports/submission_classification.csv  M2 predictions (ID,Predict) for Kaggle
  reports/submission_regression.csv      M4 predictions (ID,Predict) for Kaggle
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_fcnn.config import abspath, load_config  # noqa: E402
from gnss_fcnn.data import loader  # noqa: E402
from gnss_fcnn.evaluation import positioning  # noqa: E402
from gnss_fcnn.evaluation.metrics import classification_metrics  # noqa: E402
from gnss_fcnn.training.run import fit_model  # noqa: E402

VARIANTS = ["fcnn_m1", "fcnn_m2", "fcnn_m3", "fcnn_m4", "fcnn_m5"]
PAPER_T8 = {  # train_acc, val_acc, val_rmse
    "M1": (0.815, 0.758, 15.16), "M2": (0.820, 0.771, 15.55), "M3": (0.816, 0.766, 15.35),
    "M4": (0.814, 0.768, 15.14), "M5": (0.812, 0.761, 15.51),
}
CFG = load_config()
TABLES = abspath(CFG, CFG.eval.report.tables_dir)
REPORTS = abspath(CFG, CFG.paths.reports_dir)
TABLES.mkdir(parents=True, exist_ok=True)

_cache: dict = {}


def run(model: str, task: str, seed: int):
    key = (model, task, seed)
    if key not in _cache:
        print(f"  training {model} {task} seed={seed} ...", flush=True)
        _cache[key] = fit_model(model=model, task=task, seed=seed)
    return _cache[key]


def _to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body]) + "\n"


def write_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TABLES / f"{name}.csv", index=False)
    (TABLES / f"{name}.md").write_text(_to_md(df))


# ---------------------------------------------------------------- Table 8 ----
def table8() -> pd.DataFrame:
    print("[Table 8] 5 variants x both tasks (seed 42)")
    rows = []
    for m in VARIANTS:
        _, _, _, c, _ = run(m, "classification", 42)
        _, _, _, r, _ = run(m, "regression", 42)
        name = c["model"]
        pt = PAPER_T8[name]
        rows.append({
            "Variant": name,
            "TrainAcc": round(c["train_metric"], 3), "TrainAcc_paper": pt[0],
            "ValAcc": round(c["val_metric"], 3), "ValAcc_paper": pt[1],
            "ValRMSE_m": round(r["val_metric"], 2), "ValRMSE_paper": pt[2],
        })
    df = pd.DataFrame(rows)
    write_table(df, "table8")
    best_clf = df.loc[df.ValAcc.idxmax(), "Variant"]
    best_reg = df.loc[df.ValRMSE_m.idxmin(), "Variant"]
    print(f"  -> best clf = {best_clf} (paper M2), best reg = {best_reg} (paper M4)")
    return df


# ---------------------------------------------------------- robustness -------
def robustness() -> pd.DataFrame:
    rob = CFG.eval.robustness
    seeds = [int(rob.base_seed) + i for i in range(int(rob.n_seeds))]
    print(f"[Robustness] M2/M4 over {len(seeds)} seeds {seeds[0]}..{seeds[-1]}")
    acc = [run("fcnn_m2", "classification", s)[3]["val_metric"] for s in seeds]
    rmse = [run("fcnn_m4", "regression", s)[3]["val_metric"] for s in seeds]
    df = pd.DataFrame([
        {"Model": "M2", "Task": "classification", "Metric": "val_accuracy",
         "Mean": round(np.mean(acc), 4), "Std": round(np.std(acc), 4),
         "Min": round(min(acc), 4), "Max": round(max(acc), 4),
         "Paper": 0.771, "n_seeds": len(seeds)},
        {"Model": "M4", "Task": "regression", "Metric": "val_rmse_m",
         "Mean": round(np.mean(rmse), 3), "Std": round(np.std(rmse), 3),
         "Min": round(min(rmse), 3), "Max": round(max(rmse), 3),
         "Paper": 15.14, "n_seeds": len(seeds)},
    ])
    write_table(df, "robustness")
    print(f"  -> M2 val acc {df.iloc[0].Mean}+/-{df.iloc[0].Std} | "
          f"M4 val RMSE {df.iloc[1].Mean}+/-{df.iloc[1].Std} m")
    return df


# ------------------------------------------------------------- Table 9 -------
def table9() -> pd.DataFrame:
    print("[Table 9] selected models (val) + OLS positioning")
    _, tclf, dclf, sclf, _ = run("fcnn_m2", "classification", 42)
    _, treg, dreg, sreg, _ = run("fcnn_m4", "regression", 42)

    # full classification metrics on validation
    import torch
    with torch.no_grad():
        logits = tclf.model(torch.as_tensor(dclf["val"]["X"], dtype=torch.float32))
        prob_nlos = torch.softmax(logits, dim=1)[:, 0].numpy()
    y_val = dclf["val"]["y"]
    cm = classification_metrics(y_val, tclf.predict(dclf["val"]["X"]), prob_nlos)

    # OLS positioning baseline (Eq. 11) on validation epochs
    df_raw = loader.load_table(abspath(CFG, CFG.data.train_xlsx))
    pos = positioning.ols_validation_rmse(
        df_raw, val_fraction=float(CFG.data.split.val_fraction), seed=int(CFG.seed))

    df = pd.DataFrame([
        {"Task": "LOS/NLOS classification", "Model": "M2", "Metric": "Accuracy",
         "Val_ours": round(cm["accuracy"], 3), "Val_paper": 0.771, "Test_paper": 0.853},
        {"Task": "Pseudorange error", "Model": "M4", "Metric": "RMSE (m)",
         "Val_ours": round(sreg["val_metric"], 2), "Val_paper": 15.14, "Test_paper": 13.01},
        {"Task": "Positioning error", "Model": "OLS", "Metric": "RMSE E/N (m)",
         "Val_ours": round(pos["rmse_val"], 2), "Val_paper": 17.76, "Test_paper": 21.49},
    ])
    write_table(df, "table9")
    print(f"  -> clf acc={cm['accuracy']:.3f} (P/R/F1_NLOS="
          f"{cm['precision_nlos']:.2f}/{cm['recall_nlos']:.2f}/{cm['f1_nlos']:.2f}, "
          f"AUC={cm['roc_auc']:.3f}) | reg RMSE={sreg['val_metric']:.2f} | "
          f"OLS E/N RMSE={pos['rmse_val']:.2f}")
    return df, tclf, dclf, treg, dreg


# ---------------------------------------------------------- submissions ------
def submissions(tclf, dclf, treg, dreg) -> None:
    print("[Submission] writing Kaggle ID,Predict files (file-order locked)")
    clf_pred = tclf.predict(dclf["test"]["X"]).astype(int)
    pd.DataFrame({"ID": np.arange(len(clf_pred)), "Predict": clf_pred}).to_csv(
        REPORTS / "submission_classification.csv", index=False)
    reg_pred = treg.predict(dreg["test"]["X"]).astype(float)
    pd.DataFrame({"ID": np.arange(len(reg_pred)), "Predict": np.round(reg_pred, 4)}).to_csv(
        REPORTS / "submission_regression.csv", index=False)
    print(f"  -> classification: {len(clf_pred)} rows "
          f"(NLOS={int((clf_pred==0).sum())}, LOS={int((clf_pred==1).sum())})")
    print(f"  -> regression: {len(reg_pred)} rows, pred range "
          f"[{reg_pred.min():.1f}, {reg_pred.max():.1f}] m")


def main() -> None:
    table8()
    robustness()
    df9, tclf, dclf, treg, dreg = table9()
    submissions(tclf, dclf, treg, dreg)
    print("\nStage 04 complete -> reports/tables/ + reports/submission_*.csv")


if __name__ == "__main__":
    main()
