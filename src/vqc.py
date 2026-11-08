"""
VQC / VQC + data re-uploading (PennyLane + PyTorch)

ind16 노트북의 VQC 셀을 옮겼다.
입력은 QSVM과 같은 인코딩을 쓴다 (src.qsvm.encode).
임베딩: angle(RY) / iqp(IQPEmbedding, 인접 큐비트) / dense_angle(RY + RZ) / amplitude(AmplitudeEmbedding).
amplitude 는 회로 맨 앞에서만 쓸 수 있어서 data re-uploading 에는 쓰지 않는다.
큐비트 수 n을 인자로 받아 1~16큐비트를 반복할 수 있게 했다.
"""

import copy
import random
import time

import numpy as np
import pandas as pd

import pennylane as qml
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score

from config import (
    SEED,
    THRESHOLD,
    N_LAYERS,
    BATCH_SIZE,
    MAX_EPOCHS,
    PATIENCE,
    LEARNING_RATE,
    N_REUPLOADS,
    ENCODING,
    REDUCTION,
    QSVM_REPS,
)
from src.metrics import compute_metrics, print_report
from src.qsvm import encode, NotApplicable


# 소규모 시뮬레이터 실험이므로 CPU 사용
DEVICE = torch.device("cpu")
DTYPE = torch.float64


# ============================================================
# 1. 텐서 변환 / 예측
# ============================================================

def make_feature_tensor(array):
    return torch.as_tensor(
        np.asarray(array, dtype=np.float64),
        dtype=DTYPE,
        device=DEVICE,
    )


def make_target_tensor(labels):
    return torch.as_tensor(
        np.asarray(labels, dtype=np.float64).reshape(-1),
        dtype=DTYPE,
        device=DEVICE,
    )


def predict_logits(model, features, batch_size=128):
    model.eval()

    feature_tensor = make_feature_tensor(features)
    outputs = []

    with torch.no_grad():
        for batch in feature_tensor.split(batch_size):
            outputs.append(model(batch))

    return torch.cat(outputs, dim=0)


def predict_probability(model, features):
    logits = predict_logits(model, features)
    return torch.sigmoid(logits).cpu().numpy()


def evaluate(model, angles, y, name):
    """train / valid / test 성능표와 테스트 분류 보고서"""

    probabilities = {
        split: predict_probability(model, features)
        for split, features in zip(["train", "valid", "test"], angles)
    }

    results = pd.DataFrame({
        split: compute_metrics(labels, probabilities[split], THRESHOLD)
        for split, labels in zip(["train", "valid", "test"], y)
    }).T

    print(f"\n[{name} 성능]")
    print(results.round(4).to_string())

    print_report(y[2], probabilities["test"], name, threshold=THRESHOLD)

    return results, probabilities


# ============================================================
# 2. 데이터 임베딩 (인코딩별 회로 앞부분)
# ============================================================

def embed(inputs, n_qubits, embedding="angle"):
    wires = range(n_qubits)

    if embedding == "angle":
        # 입력값을 각 큐비트의 RY 회전각으로 인코딩 (기존 방식)
        qml.AngleEmbedding(features=inputs, wires=wires, rotation="Y")

    elif embedding == "dense_angle":
        # 큐비트 하나에 변수 2개: RY(앞 n개) 후 RZ(뒤 n개)
        qml.AngleEmbedding(features=inputs[..., :n_qubits], wires=wires, rotation="Y")
        qml.AngleEmbedding(features=inputs[..., n_qubits:], wires=wires, rotation="Z")

    elif embedding == "iqp":
        # QSVM 특징 맵과 같은 구조: H → RZ(x_i), 인접 큐비트 ZZ(x_i x_{i+1}) 반복
        pattern = [[i, i + 1] for i in range(n_qubits - 1)]
        qml.IQPEmbedding(
            features=inputs,
            wires=wires,
            n_repeats=QSVM_REPS,
            pattern=pattern if pattern else None,
        )

    elif embedding == "amplitude":
        qml.AmplitudeEmbedding(features=inputs, wires=wires, normalize=True, pad_with=0.0)

    else:
        raise ValueError(f"알 수 없는 임베딩: {embedding}")


# ============================================================
# 3. VQC (임베딩 + StronglyEntanglingLayers)
# ============================================================

def build_vqc(n_qubits, n_layers=N_LAYERS, embedding="angle"):
    # 입력: 기업당 n_qubits개 회전각
    # 출력: 각 큐비트의 Pauli-Z 기대값
    quantum_device = qml.device(
        "default.qubit",
        wires=n_qubits,
        shots=None,  # 샷 샘플링 없이 기대값 계산
    )

    @qml.qnode(
        quantum_device,
        interface="torch",
        diff_method="backprop",
    )
    def quantum_circuit(inputs, weights):
        # 입력 인코딩 (기본: 각 큐비트의 RY 회전각)
        embed(inputs, n_qubits, embedding)

        # 학습 가능한 회전 게이트 + 얽힘 게이트
        qml.StronglyEntanglingLayers(
            weights=weights,
            wires=range(n_qubits),
        )

        return [
            qml.expval(qml.PauliZ(wire))
            for wire in range(n_qubits)
        ]

    class VQCClassifier(nn.Module):
        def __init__(self, n_qubits, n_layers):
            super().__init__()

            weight_shapes = {
                "weights": qml.StronglyEntanglingLayers.shape(
                    n_layers=n_layers,
                    n_wires=n_qubits,
                )
            }

            self.quantum_layer = qml.qnn.TorchLayer(
                quantum_circuit,
                weight_shapes=weight_shapes,
            )

            # 여러 큐비트의 측정값을 하나의 분류 logit으로 결합
            self.output_layer = nn.Linear(n_qubits, 1)

        def forward(self, inputs):
            measurements = self.quantum_layer(inputs)

            # BCEWithLogitsLoss를 쓰므로 sigmoid는 여기서 적용하지 않음
            logits = self.output_layer(measurements)

            return logits.squeeze(-1)

    return VQCClassifier(
        n_qubits=n_qubits,
        n_layers=n_layers,
    ).to(device=DEVICE, dtype=DTYPE)


def train_vqc(
    angles, y, n_layers=N_LAYERS, verbose=True, seed=SEED,
    n_qubits=None, embedding="angle",
):
    """기존 'vqc(8qubit)' 훈련 셀. angles/y 는 (train, valid, test)."""

    X_train_q, X_valid_q, _ = angles
    y_train, y_valid, _ = y
    dim = X_train_q.shape[1]
    n_qubits = dim if n_qubits is None else n_qubits

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 입력 확인
    for name, features, labels in zip(["train", "valid", "test"], angles, y):
        features = np.asarray(features)
        labels = np.asarray(labels).reshape(-1)

        assert features.ndim == 2
        assert features.shape[1] == dim
        assert len(features) == len(labels)
        assert np.isfinite(features).all()
        assert np.isin(labels, [0, 1]).all()

    vqc_model = build_vqc(n_qubits, n_layers, embedding)

    if verbose:
        print(vqc_model)
        print(
            "\n학습 파라미터 수:",
            sum(
                parameter.numel()
                for parameter in vqc_model.parameters()
                if parameter.requires_grad
            ),
        )

    train_tensor = make_feature_tensor(X_train_q)
    train_target = make_target_tensor(y_train)
    valid_target = make_target_tensor(y_valid)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    train_loader = DataLoader(
        TensorDataset(train_tensor, train_target),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=loader_generator,
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        vqc_model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    best_valid_loss = float("inf")
    best_state = None
    best_epoch = 0
    waiting = 0
    history = []

    start_time = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        vqc_model.train()

        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()

            logits = vqc_model(batch_features)
            loss = criterion(logits, batch_labels)

            loss.backward()
            optimizer.step()

        # 같은 모델로 훈련·검증 손실 평가
        train_logits = predict_logits(vqc_model, X_train_q)
        valid_logits = predict_logits(vqc_model, X_valid_q)

        with torch.no_grad():
            train_loss = criterion(train_logits, train_target).item()
            valid_loss = criterion(valid_logits, valid_target).item()

        train_auc = roc_auc_score(
            y_train, torch.sigmoid(train_logits).cpu().numpy()
        )
        valid_auc = roc_auc_score(
            y_valid, torch.sigmoid(valid_logits).cpu().numpy()
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "train_auc": train_auc,
            "valid_auc": valid_auc,
        })

        if valid_loss < best_valid_loss - 1e-5:
            best_valid_loss = valid_loss
            best_state = copy.deepcopy(vqc_model.state_dict())
            best_epoch = epoch
            waiting = 0
        else:
            waiting += 1

        if verbose and (epoch == 1 or epoch % 5 == 0):
            print(
                f"Epoch {epoch:03d} | "
                f"train loss={train_loss:.4f} | "
                f"valid loss={valid_loss:.4f} | "
                f"train AUC={train_auc:.4f} | "
                f"valid AUC={valid_auc:.4f}"
            )

        if waiting >= PATIENCE:
            print(f"\n조기 종료: epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError(
            "정상적인 모델이 저장되지 않았습니다. 손실값을 확인하세요."
        )

    # 최적 검증 모델 복원
    vqc_model.load_state_dict(best_state)
    vqc_model.eval()

    training_seconds = time.perf_counter() - start_time

    print("\n최적 epoch:", best_epoch)
    print("최적 검증 loss:", round(best_valid_loss, 4))
    print("학습·검증 소요 시간(초):", round(training_seconds, 2))

    return vqc_model, {
        "history": pd.DataFrame(history),
        "best_epoch": best_epoch,
        "best_valid_loss": best_valid_loss,
        "seconds": training_seconds,
    }


# ============================================================
# 4. VQC + data re-uploading
# ============================================================

def build_vqc_reupload(
    n_qubits, n_layers=N_LAYERS, n_reuploads=N_REUPLOADS, embedding="angle",
):
    if embedding == "amplitude":
        raise NotApplicable("amplitude 임베딩은 data re-uploading 에 쓸 수 없습니다.")

    qdev = qml.device(
        "default.qubit",
        wires=n_qubits,
        shots=None,
    )

    @qml.qnode(
        qdev,
        interface="torch",
        diff_method="backprop",
    )
    def reuploading_circuit(inputs, weights):
        wires = range(n_qubits)

        # 데이터를 n_reuploads회 인코딩
        for r in range(n_reuploads):
            # 입력 인코딩 (기본: Y축 회전)
            embed(inputs, n_qubits, embedding)

            # 각 인코딩 뒤에 학습 가능한 양자 회로 적용
            qml.StronglyEntanglingLayers(
                weights=weights[r],
                wires=wires,
            )

        # 각 큐비트의 Pauli-Z 기대값
        return [
            qml.expval(qml.PauliZ(i))
            for i in wires
        ]

    weight_shapes = {
        "weights": (
            n_reuploads,
            n_layers,
            n_qubits,
            3,
        )
    }

    # 기존 셀에 있던 사용하지 않는 TorchLayer.
    # 만들 때 torch 난수를 소비하므로, 기존 코드와 같은 초기값을 쓰도록 남겨 둔다.
    _unused_quantum_layer = qml.qnn.TorchLayer(
        reuploading_circuit,
        weight_shapes,
    )

    class VQCReuploadClassifier(nn.Module):
        def __init__(self, n_qubits):
            super().__init__()

            self.quantum_layer = qml.qnn.TorchLayer(
                reuploading_circuit,
                weight_shapes,
            )

            # 양자 측정값 n개를 받아 이진분류 logit 출력
            self.classifier = nn.Linear(n_qubits, 1)

        def forward(self, x):
            quantum_output = self.quantum_layer(x)
            logits = self.classifier(quantum_output)
            return logits.squeeze(-1)

    return VQCReuploadClassifier(n_qubits).to(device=DEVICE, dtype=DTYPE)


def train_vqc_reupload(
    angles, y, verbose=True, seed=SEED, n_qubits=None, embedding="angle",
):
    """기존 'vqc(8qubit + drc)' 셀의 학습 부분. angles/y 는 (train, valid, test)."""

    X_train_q, X_valid_q, _ = angles
    y_train, y_valid, _ = y
    n_qubits = X_train_q.shape[1] if n_qubits is None else n_qubits

    torch.manual_seed(seed)
    np.random.seed(seed)

    vqc_reupload = build_vqc_reupload(n_qubits, embedding=embedding)

    Xtr = make_feature_tensor(X_train_q)
    Xva = make_feature_tensor(X_valid_q)
    ytr = make_target_tensor(y_train)
    yva = make_target_tensor(y_valid)

    train_loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    valid_loader = DataLoader(
        TensorDataset(Xva, yva),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        vqc_reupload.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    start_time = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        # ----- Training -----
        vqc_reupload.train()

        train_loss_sum = 0.0
        train_count = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()

            logits = vqc_reupload(xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(yb)
            train_count += len(yb)

        train_loss = train_loss_sum / train_count

        # ----- Validation -----
        vqc_reupload.eval()

        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for xb, yb in valid_loader:
                logits = vqc_reupload(xb)
                loss = criterion(logits, yb)

                val_loss_sum += loss.item() * len(yb)
                val_count += len(yb)

        val_loss = val_loss_sum / val_count

        # ----- Early Stopping -----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(vqc_reupload.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(
                f"Epoch {epoch + 1:03d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Valid Loss: {val_loss:.4f}"
            )

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # 가장 성능이 좋았던 검증 모델 복원
    if best_state is not None:
        vqc_reupload.load_state_dict(best_state)

    vqc_reupload.eval()

    training_seconds = time.perf_counter() - start_time

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print("학습·검증 소요 시간(초):", round(training_seconds, 2))

    return vqc_reupload, {
        "best_valid_loss": best_val_loss,
        "seconds": training_seconds,
    }


# ============================================================
# 5. 큐비트 수 반복 실험 (1 ~ 16큐비트)
#
# X = (X_train, X_valid, X_test), y = (y_train, y_valid, y_test)
# ============================================================

def _sweep(qubits, X, y, train_fn, model_name, verbose, encoding, reduction, seed):
    y = tuple(np.asarray(part).reshape(-1) for part in y)
    runs, rows = {}, {}

    for n in qubits:
        print("\n" + "=" * 70)
        print(f"{model_name} {n}qubit [{encoding}, {reduction}]")
        print("=" * 70)

        try:
            enc = encode(
                n, *X, *y, method=encoding, reduction=reduction,
                verbose=False, seed=seed,
            )
            print(f"입력: {enc.description}\n")

            model, info = train_fn(
                enc.angles, y, verbose=verbose, seed=seed,
                n_qubits=enc.n_qubits, embedding=enc.embedding_vqc,
            )
        except NotApplicable as error:
            print(f"건너뜀: {error}")
            continue

        results, probabilities = evaluate(
            model, enc.angles, y, f"{model_name} {n}qubit"
        )

        runs[n] = {
            "results": results,
            "info": info,
            "probabilities": probabilities,
            "encoding": enc,
        }
        rows[n] = {
            **results.loc["test"].to_dict(),
            "seconds": round(info["seconds"], 1),
        }

    summary = pd.DataFrame.from_dict(rows, orient="index")
    summary.index.name = "n_qubits"

    return summary, runs


def vqc_sweep(
    qubits, X, y, verbose=False, encoding=ENCODING, reduction=REDUCTION, seed=SEED,
):
    return _sweep(qubits, X, y, train_vqc, "VQC", verbose, encoding, reduction, seed)


def vqc_reupload_sweep(
    qubits, X, y, verbose=False, encoding=ENCODING, reduction=REDUCTION, seed=SEED,
):
    return _sweep(
        qubits, X, y, train_vqc_reupload, "VQC+drc", verbose, encoding, reduction, seed,
    )
