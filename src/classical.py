"""
고전 모델: XGBoost · Random Forest · SVM(RBF) · 로지스틱 회귀

입력은 선택된 변수 16개 그대로다 (PCA 없음, 이미 극단값 제한 / 대치 / RobustScaler 완료).
하이퍼파라미터는 config 의 탐색 범위(그리드)에서 검증셋 ROC-AUC가 가장 높은 조합을 고르고,
테스트셋은 최종 평가에만 쓴다. 고른 값이 탐색 범위의 끝이면 경고를 출력한다.
"""

from itertools import product
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from config import (
    SEED,
    THRESHOLD,
    XGB_PARAMS,
    XGB_GRID,
    RF_GRID,
    SVM_GRID,
    LOGISTIC_GRID,
)
from src.metrics import metrics_table


SPLITS = ["train", "valid", "test"]


def _evaluate(model, X, Y):
    probabilities = {
        split: model.predict_proba(part)[:, 1]
        for split, part in zip(SPLITS, X)
    }
    targets = dict(zip(SPLITS, Y))

    return metrics_table(targets, probabilities, threshold=THRESHOLD)


def _grid(grid):
    keys = list(grid)
    return [dict(zip(keys, values)) for values in product(*grid.values())]


# 더 줄일 수 없는 값 (이 값이 골라져도 범위를 넓힐 수 없으므로 경고하지 않음)
NATURAL_LIMITS = {"min_samples_leaf": 1, "min_child_weight": 1}


def _edge_warning(name, grid, params):
    """고른 숫자 값이 탐색 범위의 최솟값/최댓값이면 경고"""

    for key, options in grid.items():
        numbers = [v for v in options if isinstance(v, (int, float)) and not isinstance(v, bool)]

        if len(numbers) < 2 or params.get(key) not in numbers:
            continue

        if NATURAL_LIMITS.get(key) == params[key]:
            continue

        if params[key] in (min(numbers), max(numbers)):
            print(
                f"    ※ {name}: {key}={params[key]} 이(가) 탐색 범위의 끝입니다. "
                f"config 의 범위를 넓혀 확인해 보세요."
            )


def _tune(name, make_model, grid, X, Y, score=None):
    """grid 의 조합마다 훈련셋으로 학습 → 검증셋 AUC 가 가장 높은 설정"""

    score = score or (lambda model, x: model.predict_proba(x)[:, 1])
    best = None

    for params in _grid(grid):
        model = make_model(params).fit(X[0], Y[0])
        auc = roc_auc_score(Y[1], score(model, X[1]))

        if best is None or auc > best[0]:
            best = (auc, params, model)

    _edge_warning(name, grid, best[1])

    return best


# ============================================================
# 모델별 학습
# ============================================================

def train_xgb(X, Y, seed=SEED):
    """XGBoost: config.XGB_GRID 탐색, 조합마다 트리 개수는 검증셋 조기 종료"""

    best = None

    for params in _grid(XGB_GRID):
        model = XGBClassifier(**{**XGB_PARAMS, **params, "random_state": seed})
        model.fit(
            X[0], Y[0],
            eval_set=[(X[0], Y[0]), (X[1], Y[1])],
            verbose=False,
        )

        auc = roc_auc_score(Y[1], model.predict_proba(X[1])[:, 1])

        if best is None or auc > best[0]:
            best = (auc, params, model)

    _, params, model = best
    _edge_warning("XGBoost", XGB_GRID, params)

    return model, {**params, "best_iteration": int(model.best_iteration)}


def train_random_forest(X, Y, seed=SEED):
    """Random Forest: config.RF_GRID 에서 검증셋 AUC로 선택"""

    def make(params):
        return RandomForestClassifier(**params, random_state=seed, n_jobs=-1)

    _, params, model = _tune("Random Forest", make, RF_GRID, X, Y)

    return model, params


def train_svm(X, Y, seed=SEED):
    """SVM(RBF): 표준화 후 config.SVM_GRID 에서 검증셋 AUC로 선택"""

    def make(params, probability=False):
        return make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", **params, probability=probability, random_state=seed),
        )

    # 탐색은 decision_function 으로 (확률 보정 없이 빠르게)
    _, params, _ = _tune(
        "SVM", make, SVM_GRID, X, Y,
        score=lambda model, x: model.decision_function(x),
    )

    # 최종 모델만 확률 보정
    model = make(params, probability=True).fit(X[0], Y[0])

    return model, params


def train_logistic(X, Y, seed=SEED):
    """로지스틱 회귀: 표준화 후 config.LOGISTIC_GRID (C, L1/L2) 에서 검증셋 AUC로 선택"""

    def make(params):
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                **params, solver="liblinear", max_iter=5000, random_state=seed,
            ),
        )

    _, params, model = _tune("로지스틱 회귀", make, LOGISTIC_GRID, X, Y)

    return model, params


CLASSICAL_MODELS = {
    "XGBoost": train_xgb,
    "Random Forest": train_random_forest,
    "SVM (RBF)": train_svm,
    "로지스틱 회귀": train_logistic,
}


def run_classical_models(X, Y, seed=SEED, models=None):
    """
    X = (X_train, X_valid, X_test), Y = (y_train, y_valid, y_test)
    반환: {모델 이름: (train/valid/test 성능표, 선택된 설정)}
    """

    results = {}

    for name in models or CLASSICAL_MODELS:
        started = time.time()

        model, params = CLASSICAL_MODELS[name](X, Y, seed=seed)
        table = _evaluate(model, X, Y)

        results[name] = (table, params)

        print(
            f"  {name:<14} valid AUC = {table.loc['valid', 'ROC-AUC']:.4f}  "
            f"test AUC = {table.loc['test', 'ROC-AUC']:.4f}  "
            f"{params}  ({time.time() - started:.1f}초)"
        )

    return results
