"""
성능 지표 · 분류 보고서 출력 · 결과 CSV 저장

기존 노트북의 calculate_metrics / qsvm_metrics / evaluate_binary_classifier /
evaluate_xgb_pca 는 같은 8개 지표를 계산하므로 compute_metrics 하나로 합쳤다.
"""

from datetime import datetime
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    classification_report,
    confusion_matrix,
)

from config import RESULTS_CSV, THRESHOLD


METRIC_NAMES = [
    "Accuracy",
    "Balanced accuracy",
    "Precision_code2",
    "Recall_code2",
    "F1_code2",
    "ROC-AUC",
    "Average precision",
    "Log loss",
]


# ============================================================
# 1. 성능 지표
# ============================================================

def compute_metrics(labels, probabilities, threshold=THRESHOLD):
    labels = np.asarray(labels).reshape(-1)
    probabilities = np.asarray(probabilities).reshape(-1)

    if len(labels) != len(probabilities):
        raise ValueError("라벨과 예측확률의 길이가 다릅니다.")

    if not np.isfinite(probabilities).all():
        raise ValueError("예측확률에 NaN 또는 무한대가 있습니다.")

    predictions = (probabilities >= threshold).astype(int)

    return {
        "Accuracy": accuracy_score(labels, predictions),
        "Balanced accuracy": balanced_accuracy_score(labels, predictions),
        "Precision_code2": precision_score(
            labels, predictions, zero_division=0
        ),
        "Recall_code2": recall_score(
            labels, predictions, zero_division=0
        ),
        "F1_code2": f1_score(labels, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(labels, probabilities),
        "Average precision": average_precision_score(
            labels, probabilities
        ),
        "Log loss": log_loss(labels, probabilities, labels=[0, 1]),
    }


def metrics_table(labels, probabilities, threshold=THRESHOLD):
    """{split: labels}, {split: probabilities} → 행이 split인 성능표"""

    return pd.DataFrame({
        split: compute_metrics(labels[split], probabilities[split], threshold)
        for split in probabilities
    }).T


# ============================================================
# 2. 분류 보고서 / 혼동행렬 출력
# ============================================================

def print_report(
    labels, probabilities, title="", threshold=THRESHOLD,
    classification=True,
):
    labels = np.asarray(labels).reshape(-1)
    predictions = (
        np.asarray(probabilities).reshape(-1) >= threshold
    ).astype(int)

    prefix = f"{title} " if title else ""

    if classification:
        print(f"\n[{prefix}테스트 분류 보고서]")
        print(
            classification_report(
                labels,
                predictions,
                labels=[0, 1],
                target_names=["원본 코드 1", "원본 코드 2"],
                digits=4,
                zero_division=0,
            )
        )

    print(f"\n[{prefix}테스트 혼동행렬]")
    print(
        pd.DataFrame(
            confusion_matrix(labels, predictions, labels=[0, 1]),
            index=["실제 원본코드1", "실제 원본코드2"],
            columns=["예측 원본코드1", "예측 원본코드2"],
        )
    )

    return predictions


# ============================================================
# 3. 결과 CSV (results/results.csv 하나에 모든 실험 결과)
# ============================================================

KEY_COLUMNS = ["notebook", "model", "split"]


def save_result(notebook, model, results, params=None, path=RESULTS_CSV):
    """
    results: 행이 split(train/valid/test)인 DataFrame,
             또는 테스트 지표만 담은 Series/dict
    같은 (notebook, model, split) 행은 새 값으로 바꾼다.
    """

    if isinstance(results, pd.DataFrame):
        table = results.copy()
    else:
        table = pd.DataFrame([pd.Series(results)], index=["test"])

    table = table.reindex(columns=METRIC_NAMES)
    table.index.name = "split"
    table = table.reset_index()

    table.insert(0, "model", model)
    table.insert(0, "notebook", notebook)
    table["params"] = json.dumps(params or {}, ensure_ascii=False, default=str)
    table["saved_at"] = datetime.now().isoformat(timespec="seconds")

    if path.exists():
        old = pd.read_csv(path, encoding="utf-8-sig")
        keys = table[KEY_COLUMNS].apply(tuple, axis=1)
        old = old.loc[~old[KEY_COLUMNS].apply(tuple, axis=1).isin(keys)]
        table = pd.concat([old, table], ignore_index=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, encoding="utf-8-sig")

    return table


def load_results(split="test", exclude_notebook=None, path=RESULTS_CSV):
    """저장된 결과를 모델별 성능표로 읽는다 (다른 노트북 결과 비교용)."""

    if not path.exists():
        print(f"아직 저장된 결과가 없습니다: {path}")
        return pd.DataFrame(columns=METRIC_NAMES)

    table = pd.read_csv(path, encoding="utf-8-sig")
    table = table.loc[table["split"].eq(split)]

    if exclude_notebook is not None:
        table = table.loc[table["notebook"].ne(exclude_notebook)]

    return (
        table
        .assign(name=lambda t: t["model"] + " (" + t["notebook"] + ")")
        .set_index("name")[METRIC_NAMES]
    )
