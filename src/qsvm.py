"""
QSVM: 입력 인코딩 · 양자 커널 · QSVM · KTA

입력 인코딩은 encode(n_qubits, ..., method=..., reduction=...) 하나로 만든다.
  method    : basic / angle / iqp / dense_angle / amplitude  (2절 ENCODINGS 참고)
  reduction : pca / select  (큐비트 수에 맞게 입력 차원을 줄이는 방법)

basic + pca 는 기존 방식과 같다.
  - n < 16 : 변수 16개 → 표준화 → PCA n개 → 성분 표준화 → θ = π·tanh(z/2) → IQP 특징 맵
  - n = 16 : 변수 16개 → 표준화 → θ → IQP 특징 맵
"""

from types import SimpleNamespace
import time

import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from config import (
    SEED,
    FEATURES_16,
    ANGLE_SCALE,
    ENCODING,
    REDUCTION,
    C_GRID,
    QSVM_REPS,
    CLASSICAL_GAMMAS,
)
from src.metrics import compute_metrics, print_report


# ============================================================
# 1. 양자회로 입력각 변환
#
# theta = pi * tanh(z / ANGLE_SCALE)
#
# 단순 선형 각도 변환에서 큰 값이 반복적인 회전으로
# 겹치는 문제를 줄이기 위한 유계 변환.
# 단, 극단 영역의 차이가 압축되는 단점은 있음.
# ============================================================

def to_quantum_angles(z, scale=ANGLE_SCALE):
    if scale <= 0:
        raise ValueError("scale은 양수여야 합니다.")

    return np.pi * np.tanh(
        np.asarray(z, dtype=np.float64) / scale
    )


# ============================================================
# 2. 인코딩 정의
#
# 인코딩 = (차원 축소) + (양자 임베딩)
#
#   차원 축소 (config.REDUCTION)
#     "pca"    : 입력 변수 → 표준화 → PCA d개 성분 → 성분 표준화
#     "select" : 변수 선택 순위 상위 d개 → 표준화
#     (d = 입력 변수 수 이면 축소 없이 표준화만 한다)
#
#   양자 임베딩 (변수 → 양자 상태)
#     "iqp"         : 각도 → H, Z·ZZ 위상 반복 (인접 큐비트 얽힘)      d = n
#     "angle"       : 각도 → 큐비트마다 RY(θ), 얽힘 없음              d = n
#     "dense_angle" : 각도 2개 → 큐비트마다 RY(θ₁) 후 RZ(θ₂)          d = 2n  (n ≤ 8)
#     "amplitude"   : 실수 벡터를 정규화해 상태 진폭으로 사용          d = 2^n (n ≤ 4)
#
#   "basic" 은 지금까지 쓰던 조합이다: QSVM 은 iqp, VQC 는 angle(RY).
# ============================================================

class NotApplicable(Exception):
    """이 큐비트 수에서는 쓸 수 없는 인코딩 (반복 실험에서 건너뛴다)"""


ENCODINGS = {
    #  이름          : (입력 차원 d,          QSVM 임베딩,    VQC 임베딩,     각도 변환)
    "basic":        (lambda n: n,          "iqp",          "angle",        True),
    "angle":        (lambda n: n,          "angle",        "angle",        True),
    "iqp":          (lambda n: n,          "iqp",          "iqp",          True),
    "dense_angle":  (lambda n: 2 * n,      "dense_angle",  "dense_angle",  True),
    "amplitude":    (lambda n: 2 ** n,     "amplitude",    "amplitude",    False),
}


def _input_arrays(X_train, X_valid, X_test, features):
    """입력 확인 (이미 극단값 제한 / 대치 / RobustScaler가 끝난 변수)"""

    train_array = np.asarray(X_train[features], dtype=np.float64)
    valid_array = np.asarray(X_valid[features], dtype=np.float64)
    test_array = np.asarray(X_test[features], dtype=np.float64)

    for name, array in [
        ("train", train_array),
        ("valid", valid_array),
        ("test", test_array),
    ]:
        if array.ndim != 2:
            raise ValueError(f"{name}: 2차원 배열이어야 합니다.")

        if not np.isfinite(array).all():
            raise ValueError(f"{name}: 결측 또는 무한대가 있습니다.")

    if not (
        train_array.shape[1]
        == valid_array.shape[1]
        == test_array.shape[1]
    ):
        raise ValueError("훈련·검증·테스트의 변수 수가 다릅니다.")

    return train_array, valid_array, test_array


def _reduce_pca(dim, arrays, verbose, seed=SEED):
    """변수 → PCA dim개 성분 → 성분 표준화 (기존 8큐비트 '회로 세팅' 셀)"""

    train_array, valid_array, test_array = arrays

    # 훈련셋에서 상수 변수 제거
    q_constant_filter = VarianceThreshold(threshold=0.0)

    train_selected = q_constant_filter.fit_transform(train_array)
    valid_selected = q_constant_filter.transform(valid_array)
    test_selected = q_constant_filter.transform(test_array)

    max_components = min(
        train_selected.shape[0] - 1,
        train_selected.shape[1],
    )

    if not 1 <= dim <= max_components:
        raise ValueError(f"PCA 성분 수는 1~{max_components} 사이여야 합니다.")

    # PCA 전 표준화
    # PCA에서 변수별 분산 차이의 영향을 조절하기 위한 단계.
    q_input_scaler = StandardScaler()

    train_scaled = q_input_scaler.fit_transform(train_selected)
    valid_scaled = q_input_scaler.transform(valid_selected)
    test_scaled = q_input_scaler.transform(test_selected)

    # PCA로 dim개 성분에 압축 (y를 사용하지 않는 비지도 차원 축소)
    q_pca = PCA(
        n_components=dim,
        svd_solver="full",
        random_state=seed,
    )

    Z_train = q_pca.fit_transform(train_scaled)
    Z_valid = q_pca.transform(valid_scaled)
    Z_test = q_pca.transform(test_scaled)

    if verbose:
        print("성분별 설명 분산 비율:")
        print(np.round(q_pca.explained_variance_ratio_, 4))

        print(
            "누적 설명 분산 비율:",
            round(q_pca.explained_variance_ratio_.sum(), 4),
        )

    # 설명 분산 비율이 높다고 분류 정보가 반드시 잘 보존되는 것은 아님.
    # 최종 판단은 검증 분류 성능으로 해야 함.

    # PCA 성분을 훈련셋 기준으로 표준화
    # (각 성분의 입력 크기가 지나치게 달라지는 것을 방지)
    q_component_scaler = StandardScaler()

    Z = (
        q_component_scaler.fit_transform(Z_train),
        q_component_scaler.transform(Z_valid),
        q_component_scaler.transform(Z_test),
    )

    return Z, [f"PC{i + 1}" for i in range(dim)], {
        "constant_filter": q_constant_filter,
        "input_scaler": q_input_scaler,
        "pca": q_pca,
        "component_scaler": q_component_scaler,
        "explained_variance": float(q_pca.explained_variance_ratio_.sum()),
    }


def _standardize(arrays, features):
    """변수 그대로, 표준화만 (기존 16큐비트 '회로 세팅' 셀)"""

    train_array, valid_array, test_array = arrays

    # 훈련셋에서 상수 변수 제거 (모두 남아야 입력과 1:1로 대응된다)
    constant_filter = VarianceThreshold(threshold=0.0)

    train_selected = constant_filter.fit_transform(train_array)
    valid_selected = constant_filter.transform(valid_array)
    test_selected = constant_filter.transform(test_array)

    kept_features = [
        feature
        for feature, keep in zip(features, constant_filter.get_support())
        if keep
    ]

    if train_selected.shape[1] != len(features):
        removed = sorted(set(features) - set(kept_features))
        raise ValueError(
            f"상수 변수가 제거되어 {train_selected.shape[1]}개만 남았습니다. "
            f"제거된 변수: {removed}"
        )

    # 표준화 (훈련셋 기준)
    # 변수마다 입력각의 크기가 달라지지 않도록 평균 0, 표준편차 1로 맞춘다.
    input_scaler = StandardScaler()

    Z = (
        input_scaler.fit_transform(train_selected),
        input_scaler.transform(valid_selected),
        input_scaler.transform(test_selected),
    )

    return Z, kept_features, {
        "constant_filter": constant_filter,
        "input_scaler": input_scaler,
    }


# ============================================================
# 3. 인코딩 실행 (모든 양자 모델이 이 함수로 입력을 만든다)
#
# X 의 열 = 변수 선택 결과 (순위 순서). 고전 모델도 같은 열을 쓴다.
# ============================================================

def encode(
    n_qubits, X_train, X_valid, X_test, y_train, y_valid, y_test,
    method=ENCODING, reduction=REDUCTION, scale=ANGLE_SCALE,
    verbose=True, seed=SEED,
):
    if method not in ENCODINGS:
        raise ValueError(
            f"알 수 없는 인코딩: {method}. 사용 가능: {list(ENCODINGS)}"
        )

    if reduction not in ("pca", "select"):
        raise ValueError(f"알 수 없는 차원 축소: {reduction}. 사용 가능: pca, select")

    dim_of, embedding_qsvm, embedding_vqc, use_angles = ENCODINGS[method]

    features = list(X_train.columns)
    dim = dim_of(n_qubits)

    if not 1 <= dim <= len(features):
        raise NotApplicable(
            f"[{method}] {n_qubits}큐비트는 입력 {dim}개가 필요하지만 "
            f"변수는 {len(features)}개뿐입니다."
        )

    if dim == len(features):
        arrays = _input_arrays(X_train, X_valid, X_test, features)
        Z, feature_labels, fitted = _standardize(arrays, features)
        description = f"변수 {len(features)}개 그대로 (축소 없음)"
    elif reduction == "pca":
        arrays = _input_arrays(X_train, X_valid, X_test, features)
        Z, feature_labels, fitted = _reduce_pca(dim, arrays, verbose, seed=seed)
        description = f"변수 {len(features)}개 → PCA {dim}개 성분"
    else:
        chosen = features[:dim]
        arrays = _input_arrays(X_train, X_valid, X_test, chosen)
        Z, feature_labels, fitted = _standardize(arrays, chosen)
        description = f"변수 선택 순위 상위 {dim}개"

    if use_angles:
        # theta = pi * tanh(z / ANGLE_SCALE)
        inputs = tuple(to_quantum_angles(z, scale=scale) for z in Z)
    else:
        # 진폭 인코딩: 표준화 값을 그대로 쓰고, 상태를 만들 때 정규화한다.
        inputs = tuple(np.asarray(z, dtype=np.float64) for z in Z)

    description += (
        f" → {method} (QSVM: {embedding_qsvm}, VQC: {embedding_vqc}, "
        f"{n_qubits}큐비트, 입력 {dim}개)"
    )

    X_train_q, X_valid_q, X_test_q = inputs

    # 결과 검증
    for name, array in [
        ("train", X_train_q),
        ("valid", X_valid_q),
        ("test", X_test_q),
    ]:
        assert np.isfinite(array).all()
        assert array.shape[1] == dim

        if use_angles:
            assert np.abs(array).max() <= np.pi

        if verbose:
            print(
                f"{name}: shape={array.shape}, "
                f"범위=({array.min():.4f}, {array.max():.4f})"
            )

    # 타깃은 기존 것을 그대로 사용
    assert len(X_train_q) == len(y_train)
    assert len(X_valid_q) == len(y_valid)
    assert len(X_test_q) == len(y_test)

    if verbose:
        print(f"\n{description}")
        print("입력 ↔ 변수 대응:")
        for i, label in enumerate(feature_labels):
            print(f"  input {i:2d} : {label}")

    # 데이터프레임으로 정리
    columns = [f"x_{label}" for label in feature_labels]

    frames = tuple(
        pd.DataFrame(array, index=frame.index, columns=columns)
        for array, frame in zip(inputs, (X_train, X_valid, X_test))
    )

    return SimpleNamespace(
        n_qubits=n_qubits,
        dim=dim,
        method=method,
        reduction=reduction,
        embedding_qsvm=embedding_qsvm,
        embedding_vqc=embedding_vqc,
        description=description,
        X_train_q=X_train_q,
        X_valid_q=X_valid_q,
        X_test_q=X_test_q,
        angles=inputs,
        X_train_q_df=frames[0],
        X_valid_q_df=frames[1],
        X_test_q_df=frames[2],
        feature_labels=feature_labels,
        fitted=fitted,
    )


# ============================================================
# 4. QSVM: NumPy 상태벡터 시뮬레이션
# ============================================================

# ------------------------------------------------------------
# 1. 양자 특징 맵 (IQP/ZZ 계열, 인접 큐비트 얽힘)
#
#   |psi(x)> = D(x) H^n D(x) H^n |0>
#   D(x) = exp( i [ sum_i x_i z_i + sum_i x_i x_{i+1} z_i z_{i+1} ] ),  z = +-1
# ------------------------------------------------------------

def _zmatrix(n):
    index = np.arange(2 ** n)[:, None]
    bits = (index >> np.arange(n - 1, -1, -1)) & 1

    return (1 - 2 * bits).astype(np.float32)


def _hadamard_all(state, n):
    batch = state.shape[0]

    for k in range(n):
        block = state.reshape(batch, 2 ** k, 2, 2 ** (n - k - 1))

        plus = block[:, :, 0] + block[:, :, 1]
        minus = block[:, :, 0] - block[:, :, 1]

        state = (
            np.stack((plus, minus), axis=2).reshape(batch, -1)
            * np.float32(1 / np.sqrt(2))
        )

    return state


def feature_states(features, reps=2, batch=128):
    features = np.asarray(features, dtype=np.float32)
    n = features.shape[1]

    z = _zmatrix(n)
    z_pair = z[:, :-1] * z[:, 1:]

    states = np.empty((len(features), 2 ** n), dtype=np.complex64)

    for start in range(0, len(features), batch):
        x = features[start:start + batch]

        phase = x @ z.T + (x[:, :-1] * x[:, 1:]) @ z_pair.T
        diagonal = np.exp(1j * phase).astype(np.complex64, copy=False)

        state = np.full(
            (len(x), 2 ** n),
            1 / np.sqrt(2 ** n),
            dtype=np.complex64,
        )

        for r in range(reps):
            state = state * diagonal

            if r < reps - 1:
                state = _hadamard_all(state, n)

        states[start:start + batch] = state

    return states


def fidelity_kernel(states_a, states_b_conj, chunk=256):
    """K = abs(<psi_a|psi_b>)^2"""

    kernel = np.empty(
        (len(states_a), len(states_b_conj)),
        dtype=np.float64,
    )

    for start in range(0, len(states_a), chunk):
        overlap = states_a[start:start + chunk] @ states_b_conj.T
        kernel[start:start + chunk] = np.abs(overlap) ** 2

    return kernel


# ------------------------------------------------------------
# 2. 임베딩별 양자 상태와 충실도 커널
#
#   iqp         : 위의 feature_states (2^n 상태벡터)
#   angle       : 큐비트마다 RY(θ)|0>          → 곱 상태라 큐비트별 2차원 상태만 저장
#   dense_angle : 큐비트마다 RZ(φ) RY(θ)|0>    → 곱 상태
#   amplitude   : x / ||x|| 를 그대로 상태 진폭으로 사용
#
# 곱 상태의 충실도 커널은 큐비트별 겹침의 곱이다:
#   K(a, b) = Π_i |<φ_i(a)|φ_i(b)>|²
# ------------------------------------------------------------

def _product_states(theta, phi=None):
    """(샘플, 큐비트, 2) 복소 배열: 큐비트마다 RZ(phi) RY(theta)|0>"""

    theta = np.asarray(theta, dtype=np.float64)
    cos, sin = np.cos(theta / 2), np.sin(theta / 2)

    if phi is None:
        states = np.stack([cos, sin], axis=-1).astype(np.complex128)
    else:
        phi = np.asarray(phi, dtype=np.float64)
        states = np.stack(
            [np.exp(-0.5j * phi) * cos, np.exp(0.5j * phi) * sin], axis=-1
        )

    return states


def _amplitude_states(features):
    features = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(features, axis=1, keepdims=True)

    # 모든 값이 0인 행은 |0...0> 으로 둔다
    states = np.zeros_like(features)
    states[:, 0] = 1.0
    nonzero = norms[:, 0] > 1e-12
    states[nonzero] = features[nonzero] / norms[nonzero]

    return states


def prepare_states(features, embedding="iqp", reps=2):
    if embedding == "iqp":
        return feature_states(features, reps)

    if embedding == "angle":
        return _product_states(features)

    if embedding == "dense_angle":
        n = features.shape[1] // 2
        return _product_states(features[:, :n], features[:, n:])

    if embedding == "amplitude":
        return _amplitude_states(features)

    raise ValueError(f"알 수 없는 임베딩: {embedding}")


def state_kernel(states_a, states_b_conj, embedding="iqp"):
    """K = abs(<psi_a|psi_b>)^2"""

    if embedding in ("angle", "dense_angle"):
        kernel = np.ones((len(states_a), len(states_b_conj)), dtype=np.float64)

        for qubit in range(states_a.shape[1]):
            overlap = states_a[:, qubit, :] @ states_b_conj[:, qubit, :].T
            kernel *= np.abs(overlap) ** 2

        return kernel

    return fidelity_kernel(states_a, states_b_conj)


# ------------------------------------------------------------
# 3. QSVM 학습: 각도 스케일(gamma)과 C를 검증셋 AUC로 선택
# ------------------------------------------------------------

def run_qsvm(
    angles_train, angles_valid, angles_test,
    labels_train, labels_valid, labels_test,
    gammas, reps=2, seed=SEED, embedding="iqp",
):
    best = None

    for gamma in gammas:
        started = time.time()

        states_train = prepare_states(gamma * angles_train, embedding, reps)
        states_train_conj = states_train.conj()

        kernel_train = state_kernel(states_train, states_train_conj, embedding)
        kernel_valid = state_kernel(
            prepare_states(gamma * angles_valid, embedding, reps),
            states_train_conj,
            embedding,
        )

        size = len(kernel_train)
        off_diagonal = (
            (kernel_train.sum() - np.trace(kernel_train))
            / (size * size - size)
        )

        for c in C_GRID:
            svm = SVC(kernel="precomputed", C=c).fit(
                kernel_train, labels_train
            )

            auc = roc_auc_score(
                labels_valid, svm.decision_function(kernel_valid)
            )

            print(
                f"  gamma={gamma:<5} C={c:<5} "
                f"valid AUC={auc:.4f}  "
                f"(커널 비대각 평균={off_diagonal:.4f})"
            )

            if best is None or auc > best["auc"]:
                best = {
                    "auc": auc,
                    "gamma": gamma,
                    "C": c,
                    "kernel_train": kernel_train,
                    "kernel_valid": kernel_valid,
                    "states_train_conj": states_train_conj,
                }

        print(f"  [gamma={gamma}] {time.time() - started:.0f}초")

    gamma, c = best["gamma"], best["C"]

    if len(gammas) > 1 and gamma in (gammas[0], gammas[-1]):
        print(
            "\n※ 선택된 gamma가 탐색 범위의 끝입니다. "
            "범위를 넓혀 다시 확인하세요."
        )

    if c in (C_GRID[0], C_GRID[-1]):
        print(
            "※ 선택된 C가 탐색 범위의 끝입니다. "
            "C_GRID를 넓혀 다시 확인하세요."
        )

    kernel_test = state_kernel(
        prepare_states(gamma * angles_test, embedding, reps),
        best["states_train_conj"],
        embedding,
    )

    svm = SVC(
        kernel="precomputed",
        C=c,
        probability=True,
        random_state=seed,
    ).fit(best["kernel_train"], labels_train)

    probabilities = {
        "train": svm.predict_proba(best["kernel_train"])[:, 1],
        "valid": svm.predict_proba(best["kernel_valid"])[:, 1],
        "test": svm.predict_proba(kernel_test)[:, 1],
    }

    labels = {
        "train": labels_train,
        "valid": labels_valid,
        "test": labels_test,
    }

    results = pd.DataFrame({
        split: compute_metrics(labels[split], probabilities[split])
        for split in probabilities
    }).T

    print(f"\n선택된 설정: gamma={gamma}, C={c}")

    return results, {
        "gamma": gamma,
        "C": c,
        "probabilities": probabilities,
    }


# ------------------------------------------------------------
# 4. 고전 기준선: 같은 입력에 RBF-SVM / 로지스틱 회귀
# ------------------------------------------------------------

def run_classical(
    angles_train, angles_valid, angles_test,
    labels_train, labels_valid, labels_test,
    seed=SEED,
):
    best = None

    for gamma in CLASSICAL_GAMMAS:
        for c in C_GRID:
            model = SVC(kernel="rbf", gamma=gamma, C=c).fit(
                angles_train, labels_train
            )

            auc = roc_auc_score(
                labels_valid, model.decision_function(angles_valid)
            )

            if best is None or auc > best[0]:
                best = (auc, gamma, c)

    _, gamma, c = best

    rbf = SVC(
        kernel="rbf",
        gamma=gamma,
        C=c,
        probability=True,
        random_state=seed,
    ).fit(angles_train, labels_train)

    logistic = LogisticRegression(max_iter=2000).fit(
        angles_train, labels_train
    )

    return pd.DataFrame({
        f"RBF-SVM (gamma={gamma}, C={c})": compute_metrics(
            labels_test, rbf.predict_proba(angles_test)[:, 1]
        ),
        "로지스틱 회귀": compute_metrics(
            labels_test, logistic.predict_proba(angles_test)[:, 1]
        ),
    }).T


# ============================================================
# 5. KTA
# ============================================================

# ============================================================
# KTA (Kernel-Target Alignment, 커널-타깃 정렬)
#
#   KTA(K, y) = <Kc, y yᵀ>_F / (||Kc||_F · ||y yᵀ||_F)
#
#   Kc : 중심화한 커널 행렬,  y : 라벨(+1 / -1)을 평균 0으로 맞춘 벡터
#
# 커널 행렬이 "같은 클래스끼리 비슷하고 다른 클래스끼리 다르다"는
# 정답 구조(y yᵀ)와 얼마나 닮았는지를 -1~1로 나타낸다.
# 훈련셋의 라벨만 쓰고, 모델(SVM)을 학습하지 않고도 계산할 수 있다.
# ============================================================

def centered_kta(kernel, labels):
    y = 2.0 * np.asarray(labels, dtype=np.float64).reshape(-1) - 1.0
    y = y - y.mean()

    kc = (
        kernel
        - kernel.mean(axis=0, keepdims=True)
        - kernel.mean(axis=1, keepdims=True)
        + kernel.mean()
    )

    numerator = y @ kc @ y
    denominator = np.linalg.norm(kc) * (y @ y)

    return float(numerator / denominator)


def make_subsample(angles, labels, size, seed=SEED):
    """훈련셋에서 클래스 비율을 맞춰 size개를 뽑는다 (KTA 계산용)."""

    labels = np.asarray(labels).reshape(-1)
    rng = np.random.default_rng(seed)

    chosen = []

    for value in (0, 1):
        index = np.flatnonzero(labels == value)
        take = min(size // 2, len(index))
        chosen.append(rng.choice(index, size=take, replace=False))

    chosen = np.sort(np.concatenate(chosen))

    return angles[chosen], labels[chosen]


def kernel_on(angles, scale, reps, embedding="iqp"):
    """angles * scale 을 특징 맵에 넣어 만든 충실도 커널 (샘플끼리)"""

    states = prepare_states(angles * scale, embedding, reps)

    return state_kernel(states, states.conj(), embedding)


# ------------------------------------------------------------
# 1단계: 특징 맵(gamma, reps)을 KTA로 선택
# ------------------------------------------------------------

def kta_select_feature_map(
    angles, labels, gammas, reps_list, size, seed=SEED, embedding="iqp",
):
    sub_angles, sub_labels = make_subsample(angles, labels, size, seed=seed)

    rows = []

    for reps in reps_list:
        for gamma in gammas:
            started = time.time()

            score = centered_kta(
                kernel_on(sub_angles, gamma, reps, embedding), sub_labels
            )

            rows.append({"reps": reps, "gamma": gamma, "KTA": score})

            print(
                f"  reps={reps} gamma={gamma:<6} KTA={score:.4f} "
                f"({time.time() - started:.1f}초)"
            )

    table = pd.DataFrame(rows)
    best = table.loc[table["KTA"].idxmax()]

    return table, best


# ------------------------------------------------------------
# 2단계: 특징별 가중치를 KTA로 학습
#
# 입력 특징 i의 회전각을 w_i 배 한다 (w_i 가 클수록 그 특징을 크게 반영).
# 미분 없이, 특징을 하나씩 w_i 를 키우거나 줄여 보고 KTA가 오르면 채택한다.
# ------------------------------------------------------------

def kta_learn_weights(
    angles, labels, gamma, reps, size,
    factor_schedule=((0.5, 2.0), (0.75, 1.33)),
    bounds=(0.1, 5.0),
    seed=SEED,
    embedding="iqp",
):
    sub_angles, sub_labels = make_subsample(angles, labels, size, seed=seed)

    n_features = angles.shape[1]
    weights = np.ones(n_features)

    def score_of(w):
        return centered_kta(
            kernel_on(sub_angles, gamma * w, reps, embedding), sub_labels
        )

    best = score_of(weights)
    trace = [("시작", best)]

    print(f"  시작 KTA = {best:.4f}")

    for sweep, factors in enumerate(factor_schedule, start=1):
        started = time.time()

        for i in range(n_features):
            for factor in factors:
                candidate = weights.copy()
                candidate[i] = np.clip(
                    weights[i] * factor, bounds[0], bounds[1]
                )

                if candidate[i] == weights[i]:
                    continue

                score = score_of(candidate)

                if score > best + 1e-6:
                    best = score
                    weights = candidate

        trace.append((f"{sweep}회 탐색 후", best))

        print(
            f"  {sweep}회 탐색 후 KTA = {best:.4f} "
            f"({time.time() - started:.0f}초)"
        )

    return weights, trace


def alignment_on(angles, labels, gamma, weights, reps, embedding="iqp"):
    """주어진 데이터 전체의 KTA (검증셋 점검용)"""

    return centered_kta(
        kernel_on(angles, gamma * weights, reps, embedding), labels
    )


# ============================================================
# 6. 실험 실행 (큐비트 수 하나에 대한 실행)
# ============================================================

def qsvm_experiment(
    name, angles, labels, gammas, reps=QSVM_REPS, seed=SEED, embedding="iqp",
):
    """기존 'QSVM 훈련 · 검증' 셀. angles/labels 는 (train, valid, test)."""

    print(f"[{name}: 검증셋 AUC로 gamma, C 선택]")

    results, info = run_qsvm(
        *angles, *labels, gammas=gammas, reps=reps, seed=seed, embedding=embedding,
    )

    print(f"\n[{name} 성능]")
    print(results.round(4).to_string())

    print_report(labels[2], info["probabilities"]["test"], name)

    return results, info


def classical_experiment(input_name, angles, labels, seed=SEED):
    """기존 '고전 기준선' 셀"""

    results = run_classical(*angles, *labels, seed=seed)

    print(f"[같은 입력({input_name})의 고전 모델: 테스트 성능]")
    print(results.round(4).to_string())

    return results


def kta_feature_map_experiment(
    n_qubits, angles, labels, gammas, reps_list, size, seed=SEED, embedding="iqp",
):
    """기존 'KTA로 특징 맵(gamma, reps) 선택 → 모델 A' 셀"""

    angles_train = angles[0]
    labels_train = labels[0]

    # reps 는 iqp 임베딩에서만 의미가 있다
    if embedding != "iqp":
        reps_list = (1,)

    print("[KTA로 특징 맵(gamma, reps) 선택: 훈련셋 부분표본]")

    table, best = kta_select_feature_map(
        angles_train, labels_train,
        gammas, reps_list, size, seed=seed, embedding=embedding,
    )

    print("\n[KTA 표 (행: gamma, 열: reps)]")
    print(
        table
        .pivot(index="gamma", columns="reps", values="KTA")
        .round(4)
        .to_string()
    )

    gamma = float(best["gamma"])
    reps = int(best["reps"])

    print(
        f"\n선택된 특징 맵: gamma={gamma}, reps={reps}, "
        f"KTA={best['KTA']:.4f}"
    )

    if gamma in (gammas[0], gammas[-1]):
        print("※ 선택된 gamma가 탐색 범위의 끝입니다. 범위를 넓혀 다시 확인하세요.")

    print("\n[QSVM-KTA(특징 맵 선택) 훈련: C는 검증셋 AUC로 선택]")

    results, info = run_qsvm(
        *angles, *labels,
        gammas=(gamma,),
        reps=reps,
        seed=seed,
        embedding=embedding,
    )

    print(f"\n[QSVM-KTA(특징 맵 선택) {n_qubits}큐비트 성능]")
    print(results.round(4).to_string())

    print_report(labels[2], info["probabilities"]["test"], classification=False)

    info.update(table=table, best=best, reps=reps)

    return results, info


def kta_weight_experiment(
    n_qubits, angles, labels, feature_labels, gamma, reps, size,
    seed=SEED, embedding="iqp",
):
    """기존 'KTA로 특징별 가중치 학습 → 모델 B' 셀"""

    angles_train, angles_valid, angles_test = angles
    labels_train, labels_valid, labels_test = labels

    print("[KTA로 특징별 가중치 학습: 훈련셋 부분표본]")

    weights, trace = kta_learn_weights(
        angles_train, labels_train,
        gamma, reps, size, seed=seed, embedding=embedding,
    )

    weight_table = (
        pd.DataFrame({
            "특징": feature_labels,
            "가중치": weights,
        })
        .sort_values("가중치", ascending=False)
        .reset_index(drop=True)
    )

    print("\n[학습된 특징별 가중치 (1.0 = 변화 없음, 범위 0.1~5)]")
    print(weight_table.round(3).to_string())

    # 가중치가 훈련 부분표본에만 맞춰졌는지 검증셋 KTA로 점검
    ones = np.ones(angles_train.shape[1])

    valid_kta_before = alignment_on(
        angles_valid, labels_valid, gamma, ones, reps, embedding=embedding,
    )
    valid_kta_after = alignment_on(
        angles_valid, labels_valid, gamma, weights, reps, embedding=embedding,
    )

    print(
        f"\n검증셋 KTA: 가중치 학습 전 {valid_kta_before:.4f} "
        f"→ 후 {valid_kta_after:.4f}"
    )

    print("\n[QSVM-KTA(+특징별 가중치) 훈련: C는 검증셋 AUC로 선택]")

    results, info = run_qsvm(
        angles_train * weights,
        angles_valid * weights,
        angles_test * weights,
        labels_train, labels_valid, labels_test,
        gammas=(gamma,),
        reps=reps,
        seed=seed,
        embedding=embedding,
    )

    print(f"\n[QSVM-KTA(+특징별 가중치) {n_qubits}큐비트 성능]")
    print(results.round(4).to_string())

    print_report(labels_test, info["probabilities"]["test"], classification=False)

    info.update(
        weights=weights,
        trace=trace,
        weight_table=weight_table,
        valid_kta_before=valid_kta_before,
        valid_kta_after=valid_kta_after,
        reps=reps,
    )

    return results, info


# ============================================================
# 7. 큐비트 수 반복 실험 (1 ~ 16큐비트)
#
# X = (X_train, X_valid, X_test), y = (y_train, y_valid, y_test)
# 반환: summary (행 = 큐비트 수, 열 = 테스트 지표 + 선택된 설정),
#       runs (큐비트 수별 전체 결과)
# 인코딩이 그 큐비트 수에서 쓸 수 없으면(NotApplicable) 건너뛴다.
# ============================================================

def _header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _labels(y):
    return tuple(np.asarray(part) for part in y)


def _encode_or_skip(n, X, y, encoding, reduction, seed):
    try:
        enc = encode(
            n, *X, *y, method=encoding, reduction=reduction,
            verbose=False, seed=seed,
        )
    except NotApplicable as error:
        print(f"건너뜀: {error}")
        return None

    print(f"입력: {enc.description}\n")
    return enc


def qsvm_sweep(
    qubits, X, y, gammas, reps=QSVM_REPS,
    encoding=ENCODING, reduction=REDUCTION, seed=SEED,
):
    labels = _labels(y)
    runs, rows = {}, {}

    for n in qubits:
        _header(f"QSVM {n}qubit [{encoding}, {reduction}]")
        started = time.time()

        enc = _encode_or_skip(n, X, y, encoding, reduction, seed)

        if enc is None:
            continue

        results, info = qsvm_experiment(
            f"QSVM {n}qubit", enc.angles, labels, gammas,
            reps=reps, seed=seed, embedding=enc.embedding_qsvm,
        )

        runs[n] = {"results": results, "info": info, "encoding": enc}
        rows[n] = {
            **results.loc["test"].to_dict(),
            "gamma": info["gamma"],
            "C": info["C"],
            "seconds": round(time.time() - started, 1),
        }

    summary = pd.DataFrame.from_dict(rows, orient="index")
    summary.index.name = "n_qubits"

    return summary, runs


def kta_sweep(
    qubits, X, y, gammas, reps_list, size,
    encoding=ENCODING, reduction=REDUCTION, seed=SEED,
):
    """큐비트 수마다 모델 A(특징 맵 선택)와 모델 B(+특징별 가중치)"""

    labels = _labels(y)
    runs = {}
    rows_a, rows_b = {}, {}

    for n in qubits:
        _header(f"QSVM-KTA {n}qubit [{encoding}, {reduction}]")
        started = time.time()

        enc = _encode_or_skip(n, X, y, encoding, reduction, seed)

        if enc is None:
            continue

        print(f"--- 모델 A. KTA로 특징 맵 선택 ({n}qubit) ---")
        results_a, info_a = kta_feature_map_experiment(
            n, enc.angles, labels, gammas, reps_list, size,
            seed=seed, embedding=enc.embedding_qsvm,
        )

        print(f"\n--- 모델 B. KTA로 특징별 가중치 학습 ({n}qubit) ---")
        results_b, info_b = kta_weight_experiment(
            n, enc.angles, labels, enc.feature_labels,
            info_a["gamma"], info_a["reps"], size,
            seed=seed, embedding=enc.embedding_qsvm,
        )

        runs[n] = {
            "feature_map": (results_a, info_a),
            "weights": (results_b, info_b),
            "encoding": enc,
        }
        rows_a[n] = {
            **results_a.loc["test"].to_dict(),
            "gamma": info_a["gamma"],
            "reps": info_a["reps"],
            "C": info_a["C"],
        }
        rows_b[n] = {
            **results_b.loc["test"].to_dict(),
            "gamma": info_b["gamma"],
            "reps": info_b["reps"],
            "C": info_b["C"],
            "valid KTA 전": info_b["valid_kta_before"],
            "valid KTA 후": info_b["valid_kta_after"],
            "seconds": round(time.time() - started, 1),
        }

    summary_a = pd.DataFrame.from_dict(rows_a, orient="index")
    summary_b = pd.DataFrame.from_dict(rows_b, orient="index")
    summary_a.index.name = summary_b.index.name = "n_qubits"

    return summary_a, summary_b, runs
