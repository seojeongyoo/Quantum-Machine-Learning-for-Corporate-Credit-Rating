"""
양자 부분: 입력 인코딩 · 양자 커널 · QSVM · KTA

ind16_QSVM / ind16_QSVM_KTA (그리고 ind16의 PCA 인코딩)에 똑같이 들어 있던
셀을 한 곳으로 옮겼다. 계산 로직은 원래 셀과 같다.
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
    N_QUBITS,
    N_QUBITS_16,
    ANGLE_SCALE,
    ANGLE_SCALE_16,
    C_GRID,
    QSVM_REPS,
    CLASSICAL_GAMMAS,
)
from src.metrics import compute_metrics, print_report


# ============================================================
# 1. 양자회로 입력각 변환
#
# theta = pi * tanh(z / ANGLE_SCALE)
# (기존 to_quantum_angles / to_quantum_angles16 은 같은 함수라 하나로 합침)
# ============================================================

def to_quantum_angles(z, scale=ANGLE_SCALE):
    if scale <= 0:
        raise ValueError("scale은 양수여야 합니다.")

    return np.pi * np.tanh(
        np.asarray(z, dtype=np.float64) / scale
    )


# ============================================================
# 2. 8큐비트 입력: 16개 변수 → PCA 8개 → 입력각 (기존 '회로 세팅' 셀)
# ============================================================

def encode_pca(X_train, X_valid, X_test, y_train, y_valid, y_test):
    # 16개 변수를 입력으로 사용 (이미 극단값 제한 / 대치 / RobustScaler 완료)
    X_train_xgb = X_train[FEATURES_16]
    X_valid_xgb = X_valid[FEATURES_16]
    X_test_xgb = X_test[FEATURES_16]

    # ============================================================
    # 2. 입력 확인
    # ============================================================

    train_array = np.asarray(X_train_xgb, dtype=np.float64)
    valid_array = np.asarray(X_valid_xgb, dtype=np.float64)
    test_array = np.asarray(X_test_xgb, dtype=np.float64)

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


    # ============================================================
    # 3. 훈련셋에서 상수 변수 제거
    #
    # 예: 항상 0인 결측 표시 변수
    # ============================================================

    q_constant_filter = VarianceThreshold(threshold=0.0)

    train_selected = q_constant_filter.fit_transform(train_array)
    valid_selected = q_constant_filter.transform(valid_array)
    test_selected = q_constant_filter.transform(test_array)

    max_components = min(
        train_selected.shape[0] - 1,
        train_selected.shape[1],
    )

    if not 1 <= N_QUBITS <= max_components:
        raise ValueError(
            f"N_QUBITS는 1~{max_components} 사이여야 합니다."
        )


    # ============================================================
    # 4. PCA 전 표준화
    #
    # 기존 RobustScaler가 잘못됐다는 뜻이 아님.
    # PCA에서 변수별 분산 차이의 영향을 조절하기 위한 추가 단계.
    #
    # 원핫/결측 표시처럼 희소한 이진변수도 확대될 수 있으므로
    # 이후 표준화 유무를 검증셋에서 비교할 수 있음.
    # ============================================================

    q_input_scaler = StandardScaler()

    train_scaled = q_input_scaler.fit_transform(train_selected)
    valid_scaled = q_input_scaler.transform(valid_selected)
    test_scaled = q_input_scaler.transform(test_selected)


    # ============================================================
    # 5. PCA로 소수 특징에 압축
    #
    # y를 사용하지 않는 비지도 차원 축소
    # ============================================================

    q_pca = PCA(
        n_components=N_QUBITS,
        svd_solver="full",
        random_state=SEED,
    )

    Z_train = q_pca.fit_transform(train_scaled)
    Z_valid = q_pca.transform(valid_scaled)
    Z_test = q_pca.transform(test_scaled)

    print("성분별 설명 분산 비율:")
    print(np.round(q_pca.explained_variance_ratio_, 4))

    print(
        "누적 설명 분산 비율:",
        round(q_pca.explained_variance_ratio_.sum(), 4),
    )

    # 설명 분산 비율이 높다고 분류 정보가 반드시 잘 보존되는 것은 아님.
    # 반대로 낮더라도 분류에 필요한 정보 일부를 담을 수 있으므로
    # 최종 판단은 검증 분류 성능으로 해야 함.


    # ============================================================
    # 6. PCA 성분을 훈련셋 기준으로 표준화
    #
    # 각 성분의 입력각 크기가 지나치게 달라지는 것을 방지
    # ============================================================

    q_component_scaler = StandardScaler()

    Z_train_standard = q_component_scaler.fit_transform(Z_train)
    Z_valid_standard = q_component_scaler.transform(Z_valid)
    Z_test_standard = q_component_scaler.transform(Z_test)


    # ============================================================
    # 7. 양자회로 입력각으로 변환
    #
    # theta = pi * tanh(z / ANGLE_SCALE)
    #
    # 단순 선형 각도 변환에서 큰 값이 반복적인 회전으로
    # 겹치는 문제를 줄이기 위한 유계 변환.
    # 단, 극단 영역의 차이가 압축되는 단점은 있음.
    # ============================================================


    X_train_q = to_quantum_angles(Z_train_standard)
    X_valid_q = to_quantum_angles(Z_valid_standard)
    X_test_q = to_quantum_angles(Z_test_standard)


    # ============================================================
    # 8. 데이터프레임으로 정리
    # ============================================================

    angle_columns = [
        f"theta_{i}"
        for i in range(N_QUBITS)
    ]

    X_train_q_df = pd.DataFrame(
        X_train_q,
        index=X_train.index,
        columns=angle_columns,
    )

    X_valid_q_df = pd.DataFrame(
        X_valid_q,
        index=X_valid.index,
        columns=angle_columns,
    )

    X_test_q_df = pd.DataFrame(
        X_test_q,
        index=X_test.index,
        columns=angle_columns,
    )


    # ============================================================
    # 9. 결과 검증
    # ============================================================

    for name, array in [
        ("train", X_train_q),
        ("valid", X_valid_q),
        ("test", X_test_q),
    ]:
        assert np.isfinite(array).all()
        assert array.shape[1] == N_QUBITS
        assert np.abs(array).max() <= np.pi

        print(
            f"{name}: shape={array.shape}, "
            f"각도 범위=({array.min():.4f}, {array.max():.4f})"
        )

    # 타깃은 기존 것을 그대로 사용
    assert len(X_train_q) == len(y_train)
    assert len(X_valid_q) == len(y_valid)
    assert len(X_test_q) == len(y_test)

    print("\n양자회로 입력 예시:")
    print(X_train_q_df.head())

    return SimpleNamespace(
        X_train_q=X_train_q,
        X_valid_q=X_valid_q,
        X_test_q=X_test_q,
        X_train_q_df=X_train_q_df,
        X_valid_q_df=X_valid_q_df,
        X_test_q_df=X_test_q_df,
        q_constant_filter=q_constant_filter,
        q_input_scaler=q_input_scaler,
        q_pca=q_pca,
        q_component_scaler=q_component_scaler,
    )


# ============================================================
# 3. 16큐비트 입력: 변수 16개 → 큐비트 16개, PCA 없음 (기존 '회로 세팅' 셀)
# ============================================================

def encode_direct16(X_train, X_valid, X_test, y_train, y_valid, y_test):
    # ============================================================
    # 2. 입력 확인
    #
    # 이미 극단값 제한 / 대치 / RobustScaler가 끝난 16개 변수
    # ============================================================

    train_array16 = np.asarray(X_train[FEATURES_16], dtype=np.float64)
    valid_array16 = np.asarray(X_valid[FEATURES_16], dtype=np.float64)
    test_array16 = np.asarray(X_test[FEATURES_16], dtype=np.float64)

    for name, array in [
        ("train", train_array16),
        ("valid", valid_array16),
        ("test", test_array16),
    ]:
        if array.ndim != 2:
            raise ValueError(f"{name}: 2차원 배열이어야 합니다.")

        if not np.isfinite(array).all():
            raise ValueError(f"{name}: 결측 또는 무한대가 있습니다.")


    # ============================================================
    # 3. 훈련셋에서 상수 변수 제거
    #
    # 16개가 모두 남아야 큐비트 16개와 1:1로 대응된다.
    # ============================================================

    q16_constant_filter = VarianceThreshold(threshold=0.0)

    train_selected16 = q16_constant_filter.fit_transform(train_array16)
    valid_selected16 = q16_constant_filter.transform(valid_array16)
    test_selected16 = q16_constant_filter.transform(test_array16)

    kept_features16 = [
        feature
        for feature, keep in zip(
            FEATURES_16, q16_constant_filter.get_support()
        )
        if keep
    ]

    if train_selected16.shape[1] != N_QUBITS_16:
        removed = sorted(set(FEATURES_16) - set(kept_features16))
        raise ValueError(
            f"상수 변수가 제거되어 {train_selected16.shape[1]}개만 남았습니다. "
            f"제거된 변수: {removed}"
        )


    # ============================================================
    # 4. 표준화 (훈련셋 기준)
    #
    # 변수마다 입력각의 크기가 달라지지 않도록 평균 0, 표준편차 1로 맞춘다.
    # ============================================================

    q16_input_scaler = StandardScaler()

    train_scaled16 = q16_input_scaler.fit_transform(train_selected16)
    valid_scaled16 = q16_input_scaler.transform(valid_selected16)
    test_scaled16 = q16_input_scaler.transform(test_selected16)


    # ============================================================
    # 5. 양자회로 입력각으로 변환
    #
    # theta = pi * tanh(z / ANGLE_SCALE)
    # ============================================================


    X_train_q16 = to_quantum_angles(train_scaled16, scale=ANGLE_SCALE_16)
    X_valid_q16 = to_quantum_angles(valid_scaled16, scale=ANGLE_SCALE_16)
    X_test_q16 = to_quantum_angles(test_scaled16, scale=ANGLE_SCALE_16)


    # ============================================================
    # 6. 데이터프레임으로 정리 (큐비트 i = i번째 변수)
    # ============================================================

    angle_columns16 = [f"theta_{feature}" for feature in kept_features16]

    X_train_q16_df = pd.DataFrame(
        X_train_q16, index=X_train.index, columns=angle_columns16
    )
    X_valid_q16_df = pd.DataFrame(
        X_valid_q16, index=X_valid.index, columns=angle_columns16
    )
    X_test_q16_df = pd.DataFrame(
        X_test_q16, index=X_test.index, columns=angle_columns16
    )


    # ============================================================
    # 7. 결과 검증
    # ============================================================

    for name, array in [
        ("train", X_train_q16),
        ("valid", X_valid_q16),
        ("test", X_test_q16),
    ]:
        assert np.isfinite(array).all()
        assert array.shape[1] == N_QUBITS_16
        assert np.abs(array).max() <= np.pi

        print(
            f"{name}: shape={array.shape}, "
            f"각도 범위=({array.min():.4f}, {array.max():.4f})"
        )

    assert len(X_train_q16) == len(y_train)
    assert len(X_valid_q16) == len(y_valid)
    assert len(X_test_q16) == len(y_test)

    print("\n큐비트 ↔ 변수 대응:")
    for qubit, feature in enumerate(kept_features16):
        print(f"  qubit {qubit:2d} : {feature}")

    print("\n양자회로 입력 예시:")
    print(X_train_q16_df.head())

    return SimpleNamespace(
        X_train_q16=X_train_q16,
        X_valid_q16=X_valid_q16,
        X_test_q16=X_test_q16,
        X_train_q16_df=X_train_q16_df,
        X_valid_q16_df=X_valid_q16_df,
        X_test_q16_df=X_test_q16_df,
        kept_features16=kept_features16,
        q16_constant_filter=q16_constant_filter,
        q16_input_scaler=q16_input_scaler,
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
# 3. QSVM 학습: 각도 스케일(gamma)과 C를 검증셋 AUC로 선택
# ------------------------------------------------------------

def run_qsvm(
    angles_train, angles_valid, angles_test,
    labels_train, labels_valid, labels_test,
    gammas, reps=2,
):
    best = None

    for gamma in gammas:
        started = time.time()

        states_train = feature_states(gamma * angles_train, reps)
        states_train_conj = states_train.conj()

        kernel_train = fidelity_kernel(states_train, states_train_conj)
        kernel_valid = fidelity_kernel(
            feature_states(gamma * angles_valid, reps),
            states_train_conj,
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

    kernel_test = fidelity_kernel(
        feature_states(gamma * angles_test, reps),
        best["states_train_conj"],
    )

    svm = SVC(
        kernel="precomputed",
        C=c,
        probability=True,
        random_state=SEED,
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
        random_state=SEED,
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


def kernel_on(angles, scale, reps):
    """angles * scale 을 특징 맵에 넣어 만든 충실도 커널 (샘플끼리)"""

    states = feature_states(angles * scale, reps)

    return fidelity_kernel(states, states.conj())


# ------------------------------------------------------------
# 1단계: 특징 맵(gamma, reps)을 KTA로 선택
# ------------------------------------------------------------

def kta_select_feature_map(angles, labels, gammas, reps_list, size):
    sub_angles, sub_labels = make_subsample(angles, labels, size)

    rows = []

    for reps in reps_list:
        for gamma in gammas:
            started = time.time()

            score = centered_kta(
                kernel_on(sub_angles, gamma, reps), sub_labels
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
):
    sub_angles, sub_labels = make_subsample(angles, labels, size)

    n_features = angles.shape[1]
    weights = np.ones(n_features)

    def score_of(w):
        return centered_kta(
            kernel_on(sub_angles, gamma * w, reps), sub_labels
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


def alignment_on(angles, labels, gamma, weights, reps):
    """주어진 데이터 전체의 KTA (검증셋 점검용)"""

    return centered_kta(
        kernel_on(angles, gamma * weights, reps), labels
    )


# ============================================================
# 6. 실험 실행 (8큐비트 / 16큐비트에서 똑같이 반복되던 셀)
# ============================================================

def qsvm_experiment(name, angles, labels, gammas, reps=QSVM_REPS):
    """기존 'QSVM 훈련 · 검증' 셀. angles/labels 는 (train, valid, test)."""

    print(f"[{name}: 검증셋 AUC로 gamma, C 선택]")

    results, info = run_qsvm(*angles, *labels, gammas=gammas, reps=reps)

    print(f"\n[{name} 성능]")
    print(results.round(4).to_string())

    print_report(labels[2], info["probabilities"]["test"], name)

    return results, info


def classical_experiment(input_name, angles, labels):
    """기존 '고전 기준선' 셀"""

    results = run_classical(*angles, *labels)

    print(f"[같은 입력({input_name})의 고전 모델: 테스트 성능]")
    print(results.round(4).to_string())

    return results


def kta_feature_map_experiment(n_qubits, angles, labels, gammas, reps_list, size):
    """기존 'KTA로 특징 맵(gamma, reps) 선택 → 모델 A' 셀"""

    angles_train = angles[0]
    labels_train = labels[0]

    print("[KTA로 특징 맵(gamma, reps) 선택: 훈련셋 부분표본]")

    table, best = kta_select_feature_map(
        angles_train, labels_train,
        gammas, reps_list, size,
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
    )

    print(f"\n[QSVM-KTA(특징 맵 선택) {n_qubits}큐비트 성능]")
    print(results.round(4).to_string())

    print_report(labels[2], info["probabilities"]["test"], classification=False)

    info.update(table=table, best=best, reps=reps)

    return results, info


def kta_weight_experiment(n_qubits, angles, labels, feature_labels, gamma, reps, size):
    """기존 'KTA로 특징별 가중치 학습 → 모델 B' 셀"""

    angles_train, angles_valid, angles_test = angles
    labels_train, labels_valid, labels_test = labels

    print("[KTA로 특징별 가중치 학습: 훈련셋 부분표본]")

    weights, trace = kta_learn_weights(
        angles_train, labels_train,
        gamma, reps, size,
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
        angles_valid, labels_valid, gamma, ones, reps
    )
    valid_kta_after = alignment_on(
        angles_valid, labels_valid, gamma, weights, reps
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
