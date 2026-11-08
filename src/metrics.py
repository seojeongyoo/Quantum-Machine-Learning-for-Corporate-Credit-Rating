"""
성능 지표 · 분류 보고서 출력 · 결과 CSV 저장

기존 노트북에 따로 있던 지표 함수들은 같은 8개 지표를 계산하므로
compute_metrics 하나로 합쳤다.
"""

from contextlib import contextmanager
from datetime import datetime
import os
import time
import json

import numpy as np
from scipy import stats
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

from config import RESULTS_CSV, THRESHOLD, CONFIDENCE


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
#
# 한 행 = (노트북, 모델, 변수 선택, 인코딩, 큐비트 수, seed, split)
#   인코딩 = "방법/축소" (예: basic/pca). 고전 모델은 "-"
# ============================================================

KEY_COLUMNS = ["notebook", "model", "selection", "encoding", "n_qubits", "seed", "split"]
GROUP_COLUMNS = ["notebook", "model", "selection", "encoding", "n_qubits", "split"]


def _read(path):
    table = pd.read_csv(
        path, encoding="utf-8-sig",
        dtype={"n_qubits": "Int64", "seed": "Int64"},
    )

    for column in ("selection", "encoding"):
        if column not in table:
            table[column] = "-"

    return table


def save_result(
    notebook, model, results, n_qubits=None, seed=None, params=None,
    selection="-", encoding="-", path=RESULTS_CSV,
):
    """
    results: 행이 split(train/valid/test)인 DataFrame,
             또는 테스트 지표만 담은 Series/dict
    n_qubits: 양자 모델의 큐비트 수 (고전 모델은 None)
    seed: 분할·모델 seed
    selection: 변수 선택 방법, encoding: 양자 인코딩 ("방법/축소")
    같은 (notebook, model, selection, encoding, n_qubits, seed, split) 행은 새 값으로 바꾼다.
    """

    if isinstance(results, pd.DataFrame):
        table = results.copy()
    else:
        table = pd.DataFrame([pd.Series(results)], index=["test"])

    table = table.reindex(columns=METRIC_NAMES)
    table.index.name = "split"
    table = table.reset_index()

    table.insert(0, "seed", pd.array([seed] * len(table), dtype="Int64"))
    table.insert(0, "n_qubits", pd.array([n_qubits] * len(table), dtype="Int64"))
    table.insert(0, "encoding", encoding)
    table.insert(0, "selection", selection)
    table.insert(0, "model", model)
    table.insert(0, "notebook", notebook)
    table["params"] = json.dumps(params or {}, ensure_ascii=False, default=str)
    table["saved_at"] = datetime.now().isoformat(timespec="seconds")

    path.parent.mkdir(parents=True, exist_ok=True)

    # 여러 노트북이 동시에 저장해도 행이 지워지지 않도록 잠금 파일 사용
    with _file_lock(path):
        if path.exists():
            old = _read(path)
            key = lambda t: t[KEY_COLUMNS].astype(str).apply(tuple, axis=1)
            old = old.loc[~key(old).isin(key(table))]
            table = pd.concat([old, table], ignore_index=True)

        temp = path.with_suffix(".tmp")
        table.to_csv(temp, index=False, encoding="utf-8-sig")
        _replace(temp, path)

    return table


def _replace(temp, path, attempts=60):
    """OneDrive 동기화·백신이 잠깐 파일을 잡고 있으면 거부되므로 잠시 뒤 다시 시도"""

    for attempt in range(attempts):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(1)


@contextmanager
def _file_lock(path, timeout=600):
    lock = path.with_suffix(".lock")
    started = time.time()

    while True:
        try:
            handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            # 비정상 종료로 남은 잠금은 10분 뒤 무시
            if time.time() - started > timeout:
                lock.unlink(missing_ok=True)
            time.sleep(0.2)

    try:
        yield
    finally:
        os.close(handle)
        lock.unlink(missing_ok=True)


def save_sweep(
    notebook, model, runs, seed=None, key="results", params_of=None,
    selection="-", encoding="-",
):
    """큐비트 수 반복 실험(runs[n][key] = (train/valid/test 성능표))을 저장"""

    for n, run in runs.items():
        results = run[key]
        info = None

        if isinstance(results, tuple):
            results, info = results

        params = params_of(run if info is None else info) if params_of else None
        save_result(
            notebook, model, results, n_qubits=n, seed=seed, params=params,
            selection=selection, encoding=encoding,
        )


def load_results(split="test", notebook=None, exclude_notebook=None, path=RESULTS_CSV):
    """저장된 결과를 seed별 원자료 그대로 읽는다."""

    if not path.exists():
        print(f"아직 저장된 결과가 없습니다: {path}")
        return pd.DataFrame(columns=KEY_COLUMNS + METRIC_NAMES)

    table = _read(path)

    if split is not None:
        table = table.loc[table["split"].eq(split)]

    if notebook is not None:
        table = table.loc[table["notebook"].eq(notebook)]

    if exclude_notebook is not None:
        table = table.loc[table["notebook"].ne(exclude_notebook)]

    return table.reset_index(drop=True)


# ============================================================
# 4. 여러 seed 결과 요약: 평균 · 표준편차 · 표준오차 · 신뢰구간
# ============================================================

def summarize(table, metrics=METRIC_NAMES, confidence=CONFIDENCE):
    """
    seed별 결과(load_results 의 반환값)를 (노트북, 모델, 큐비트 수, split)마다 요약한다.

    열: (지표, 통계량) — 통계량은 n(seed 수), mean, std, se, ci_low, ci_high
      se = std / sqrt(n)
      신뢰구간 = mean ± t(1 - α/2, n - 1) · se   (n = 1 이면 계산하지 않음)
    """

    if table.empty:
        return pd.DataFrame()

    groups = table.groupby(GROUP_COLUMNS, dropna=False)[list(metrics)]

    n = groups.count()
    mean = groups.mean()
    std = groups.std(ddof=1)
    se = std / np.sqrt(n)

    t_value = pd.DataFrame(
        stats.t.ppf(0.5 + confidence / 2, df=(n - 1).where(n > 1)),
        index=n.index, columns=n.columns,
    )

    parts = {
        "n": n,
        "mean": mean,
        "std": std,
        "se": se,
        "ci_low": mean - t_value * se,
        "ci_high": mean + t_value * se,
    }

    summary = pd.concat(parts, axis=1).swaplevel(0, 1, axis=1)

    order = [
        (metric, stat)
        for metric in metrics
        for stat in parts
    ]

    return summary[order]


def mean_se_table(summary, metrics=("Accuracy", "ROC-AUC", "Log loss"), digits=4):
    """요약표를 '평균 ± 표준오차' 문자열 표로 바꾼다 (보고서용)."""

    if summary.empty:
        return summary

    out = pd.DataFrame(index=summary.index)

    for metric in metrics:
        mean = summary[(metric, "mean")]
        se = summary[(metric, "se")]

        out[metric] = [
            f"{m:.{digits}f}" if pd.isna(s) else f"{m:.{digits}f} ± {s:.{digits}f}"
            for m, s in zip(mean, se)
        ]

    out["seed 수"] = summary[(metrics[0], "n")].astype(int)

    return out


def summarize_results(
    split="test", notebook=None, exclude_notebook=None,
    metrics=("Accuracy", "ROC-AUC", "Log loss"), path=RESULTS_CSV,
):
    """results.csv 를 읽어 '평균 ± 표준오차' 표로 보여 준다."""

    table = load_results(
        split=split, notebook=notebook, exclude_notebook=exclude_notebook, path=path,
    )
    summary = summarize(table)

    return mean_se_table(summary, metrics=metrics), summary
