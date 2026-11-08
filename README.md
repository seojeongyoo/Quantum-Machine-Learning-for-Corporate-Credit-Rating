# Quantum-Machine-Learning-for-Corporate-Credit-Rating

양자 머신러닝(QSVM, VQC)과 고전 모델(XGBoost, SVM, 로지스틱 회귀)로 기업의 정상/폐업을 분류하는 기업신용평가 연구 코드입니다. 양자 모델은 큐비트 수를 1부터 16까지 바꿔 가며 성능을 비교합니다.

- 데이터: FILE2_중공업 (3,140개 기업, 산업 기준 1:1 매칭)
- 독립변수: t-test와 로지스틱 회귀로 고른 재무비율 16개
- 분할: Train / Validation / Test = 60% / 20% / 20% (1,884 / 628 / 628). 층화추출을 쓰고, seed 여러 개로 분할을 바꿔 가며 반복합니다.
- 하이퍼파라미터는 검증셋으로 고르고, 테스트셋은 최종 평가에만 씁니다.

## 폴더 구조

```
├─ config.py               # 경로, SEED, 컬럼, 16개 변수, 큐비트 범위, 탐색 범위 등 모든 설정
├─ src/
│  ├─ common.py            # 데이터 읽기, EDA, 전처리, 16개 변수 추출, 분할
│  ├─ qsvm.py              # 양자 입력 인코딩, 양자 커널, QSVM, QSVM-KTA, 큐비트 수 반복 실험
│  ├─ vqc.py               # VQC, VQC + data re-uploading, 큐비트 수 반복 실험
│  ├─ classical.py         # XGBoost, Random Forest, SVM, 로지스틱 회귀
│  └─ metrics.py           # 성능 지표, 분류 보고서, 결과 CSV 저장
├─ ind16.ipynb             # 고전 모델 4종, VQC, VQC + drc
├─ ind16_QSVM.ipynb        # QSVM + 고전 기준선
├─ ind16_QSVM_KTA.ipynb    # QSVM-KTA (특징 맵 선택 / 특징별 가중치)
├─ data/                   # 원본 데이터 (GitHub에 올리지 않음)
└─ results/results.csv     # 모든 실험의 성능 결과
```

노트북은 `src/`의 함수를 불러서 실험 순서만 보여 줍니다. 공통 함수는 `src/`에 한 번만 작성되어 있습니다.

## 실행 방법

1. 원본 데이터 `FILE2_중공업.xls`를 `data/` 폴더에 넣습니다. 데이터는 공개하지 않으므로 저장소에 포함되어 있지 않습니다.
2. 필요한 패키지를 설치합니다.
   ```
   pip install numpy pandas scikit-learn xgboost xlrd openpyxl matplotlib
   pip install torch pennylane      # ind16.ipynb 의 VQC 실험에만 필요
   ```
3. 프로젝트 루트에서 노트북을 엽니다. 노트북이 `config.py`와 `src/`를 상대경로로 불러옵니다.
   - Colab에서는 노트북 첫 코드 셀의 주석을 풀어 드라이브를 마운트하고, 프로젝트 폴더로 `%cd` 합니다.
4. 설정은 `config.py`에서만 바꿉니다. 실험 조합은 아래 목록으로 정합니다.
   - `SELECTION_METHODS`: 변수 선택 방법 (기본 `["fixed16"]`)
   - `ENCODINGS`, `REDUCTIONS`: 양자 인코딩과 차원 축소 (기본 `["basic"]`, `["pca"]`)
   - `QUBIT_RANGE`: QSVM·KTA 큐비트 수 (기본 1~16). 예: `[4, 8, 12, 16]`
   - `VQC_QUBIT_RANGE`: VQC 큐비트 수 (기본 1~8). 시뮬레이션 학습이 큐비트마다 약 2배씩 느려집니다.
   - `SEEDS`: 반복 seed (기본 5개). 빠르게 확인하려면 `[42]` 하나로 줄입니다.
5. 실험 셀은 조합 하나가 끝날 때마다 결과를 `results/results.csv`에 저장합니다. 결과는 노트북, 모델, 변수 선택, 인코딩, 큐비트 수, seed, split별로 한 행씩 저장됩니다. 같은 실험을 다시 실행하면 해당 행만 새 값으로 바뀝니다.

## 변수 선택

seed마다 **그 seed의 훈련셋에서만** 변수를 고릅니다(`src/common.py`의 `select_features`). 후보는 전처리 결과 148개 열과 업력이고, 순위를 매겨 상위 16개(`N_FEATURES`)를 씁니다.

| 방법 | 순위 기준 |
|---|---|
| `fixed16` | 기존 16개 고정. 전체 데이터로 고른 목록이라 테스트 정보가 섞여 있으며, 선택 편향 크기를 재는 비교용 |
| `ttest_logit` | Welch t-test(p < 0.05) 통과 → 표준화 로지스틱 회귀 \|계수\| 순 |
| `elasticnet` | Elastic-net 로지스틱 회귀(l1_ratio 0.5, C는 훈련셋 5-fold CV) \|계수\| 순 |
| `mutual_info` | 상호정보량 순 |
| `mrmr` | mRMR: F-통계량 ÷ 이미 고른 변수와의 평균 \|상관\| (탐욕 선택) |
| `xgb` | XGBoost gain 중요도 순 |

- 지금은 기존 방식(`fixed16`)만 씁니다. 다른 방법은 `SELECTION_METHODS`에 추가하면 모든 노트북에서 같이 돕니다. 방법이 2개 이상이면 `ind16.ipynb`가 **선택 안정성**도 보여 줍니다. 선택 안정성은 seed끼리 고른 변수 집합의 평균 Jaccard 지수입니다.
- `ttest_logit`은 기존 절차를 근사한 것입니다. 기존에 쓴 정확한 기준(p값 기준, 로지스틱 단계 방식)은 확인이 필요합니다.
- 업력을 후보에서 빼려면 `SELECTION_EXCLUDE = ["업력"]`로 바꿉니다.

## 여러 seed 반복 (반복 무작위 분할 검증)

`config.SEEDS`(기본 `[42, 43, 44, 45, 46]`)의 seed마다 아래를 새로 합니다.

1. train / valid / test 분할 (60 / 20 / 20, 층화추출)
2. 전처리 fit (그 seed의 훈련셋에서만)
3. 인코딩(PCA) fit, KTA 부분표본, 모델 학습 난수

각 노트북의 마지막 셀은 `results.csv`를 읽어 모델 × 큐비트 수별로 다음 통계를 보여 줍니다.

| 통계 | 계산 |
|---|---|
| mean | seed별 테스트 성능의 평균 |
| std | 표본 표준편차 (ddof = 1) |
| se | 표준오차 = std / √(seed 수) |
| 95% 신뢰구간 | mean ± t(0.975, seed 수 − 1) · se |

보고서용 표는 `평균 ± 표준오차` 형식입니다(`src/metrics.py`의 `summarize_results`). 실행 시간은 seed 개수에 비례합니다.

## 양자 입력 인코딩

양자 모델에만 적용합니다. 고전 모델은 PCA 없이 선택된 변수 16개를 그대로 씁니다.

인코딩은 두 단계로 이루어집니다. 먼저 입력 차원을 큐비트 수에 맞추고(차원 축소), 그다음 양자 상태로 바꿉니다(임베딩).

| 인코딩 (`ENCODINGS`) | 양자 상태 | 입력 수 d (n큐비트) |
|---|---|---|
| `basic` | 기존 방식: QSVM은 IQP 특징 맵, VQC는 RY 각도 | n |
| `angle` | 큐비트마다 RY(θ), 얽힘 없음 | n |
| `iqp` | H → Z·ZZ 위상 반복, 인접 큐비트 얽힘 (Havlíček et al. 2019) | n |
| `dense_angle` | 큐비트 하나에 변수 2개: RY(θ₁) 후 RZ(θ₂) | 2n (n ≤ 8) |
| `amplitude` | 입력 벡터를 정규화해 상태 진폭으로 사용 | 2ⁿ (n ≤ 4) |

| 차원 축소 (`REDUCTIONS`) | 처리 |
|---|---|
| `pca` | 표준화 → PCA d개 성분 → 성분 표준화 (기존 방식) |
| `select` | 변수 선택 순위 상위 d개 → 표준화 |

- d = 16이면 줄이지 않고 표준화만 합니다. 각도형 인코딩은 θ = π·tanh(z/2)로 바꿉니다.
- 그 큐비트 수에서 쓸 수 없는 인코딩은 자동으로 건너뜁니다. 예를 들어 amplitude는 5큐비트 이상이면 입력이 부족합니다.
- `basic` + `pca`에서 n = 8은 이전 8큐비트(PCA 8개), n = 16은 이전 16큐비트 입력과 같습니다.
- QSVM 커널은 상태를 NumPy로 정확히 계산합니다. angle과 dense_angle은 곱 상태라서 큐비트별 겹침을 곱해 빠르게 계산합니다.
- amplitude는 회로 맨 앞에서만 쓸 수 있어서 VQC + data re-uploading에서는 건너뜁니다.
- 아직 넣지 않은 인코딩도 있습니다. basis 인코딩(이진화)은 연속형 재무비율 정보를 많이 잃고, data re-uploading은 이미 VQC + drc로 실험하고 있습니다.

## 모델

| 노트북 | 모델 | 입력 |
|---|---|---|
| ind16 | XGBoost, Random Forest, SVM(RBF), 로지스틱 회귀 | 선택된 변수 16개, PCA 없음 (하이퍼파라미터는 검증셋 AUC로 선택) |
| ind16 | VQC | n큐비트 인코딩 (n = 1~8, `VQC_QUBIT_RANGE`) |
| ind16 | VQC + data re-uploading | n큐비트 인코딩, 인코딩 2회 (n = 1~8) |
| ind16_QSVM | QSVM | n큐비트 인코딩 (n = 1~16) |
| ind16_QSVM | RBF-SVM, 로지스틱 회귀 | 선택된 변수 16개 (고전 기준선) |
| ind16_QSVM_KTA | QSVM-KTA (특징 맵 선택) | n큐비트, KTA로 γ와 반복 횟수 선택 (n = 1~16) |
| ind16_QSVM_KTA | QSVM-KTA (+특징별 가중치) | n큐비트, KTA로 변수별 회전각 가중치 학습 (n = 1~16) |

QSVM의 양자 특징 맵은 IQP/ZZ 계열입니다. 인접한 큐비트끼리 ZZ로 얽힘을 만들고, 커널은 K(a, b) = |⟨ψ(a)|ψ(b)⟩|² 입니다. 상태벡터를 NumPy로 정확히 계산하므로 별도 양자 라이브러리가 필요 없습니다.

16큐비트는 상태벡터가 65,536차원이라 오래 걸립니다. QSVM은 γ 하나당 수십 초가 걸리고, VQC는 8큐비트도 Colab에서 약 4.5분이 걸렸습니다.

## 이전 결과 (PCA 사용, 리팩터링 전, Colab 실행 기준)

1~16큐비트 실험은 아직 실행하지 않았습니다. 아래는 PCA 8개(8큐비트)와 변수 16개(16큐비트)를 쓰던 이전 실험의 테스트셋 결과입니다.

| 모델 | Accuracy | ROC-AUC | Log loss |
|---|---|---|---|
| XGBoost (변수 16개) | 0.8615 | **0.9453** | **0.3044** |
| XGBoost (PCA 8개) | 0.8232 | 0.8946 | 0.4052 |
| VQC (8큐비트, PCA) | 0.6990 | 0.7791 | 0.5619 |
| VQC + drc (8큐비트, PCA) | 0.7532 | 0.8313 | 0.5128 |
| QSVM (8큐비트, PCA) | 0.8376 | 0.8973 | 0.3983 |
| QSVM (16큐비트) | **0.8583** | 0.9268 | 0.3419 |
| QSVM-KTA (8큐비트, PCA, 특징 맵 선택) | 0.8424 | 0.8980 | 0.3978 |
| QSVM-KTA (8큐비트, PCA, +특징별 가중치) | 0.8408 | 0.9001 | 0.3961 |
| QSVM-KTA (16큐비트, 특징 맵 선택) | **0.8583** | 0.9268 | 0.3419 |
| QSVM-KTA (16큐비트, +특징별 가중치) | 0.8503 | 0.9238 | 0.3520 |

- 양자 모델 중에서는 QSVM 16큐비트가 가장 좋았습니다. 그래도 XGBoost(변수 16개)보다는 ROC-AUC가 약 0.02 낮습니다.
- VQC는 QSVM보다 성능이 크게 낮았습니다. data re-uploading을 쓰면 ROC-AUC가 0.78에서 0.83으로 올랐습니다.
- 16큐비트에서 KTA로 고른 특징 맵은 검증셋 AUC로 고른 QSVM과 같은 결과를 냈습니다.

### 해석할 때 주의할 점

- 결과는 한 번의 분할(seed 42)에서 한 번 실행한 값입니다. seed에 따른 변동은 아직 확인하지 않았습니다.
- 변수 중 업력(K1)은 산식에 기업도산일을 참조하므로 데이터 누수 가능성이 있습니다.
- 타깃 코드 1/2를 0/1로 바꿨지만, 각 코드의 의미(정상/폐업)와 판정 시점은 확인이 필요합니다.

## 진행 상황

- [x] 세 노트북의 중복 코드를 `src/common.py`, `src/qsvm.py`, `src/vqc.py`, `src/metrics.py`로 정리
- [x] 실험 설정과 seed를 `config.py`로 통일하고, 데이터 경로를 `data/` 상대경로로 변경
- [x] 노트북 3개를 `src/`를 불러 쓰는 얇은 노트북으로 변경
- [x] XGBoost + PCA 제거, 양자 모델(QSVM, QSVM-KTA, VQC, VQC + drc)을 1~16큐비트 반복 실험으로 변경 (n < 16은 PCA n개)
- [x] 여러 seed 반복 실험과 평균 ± 표준오차 / 95% 신뢰구간 요약 추가
- [x] 변수 선택 6가지(훈련셋에서만 선택)와 선택 안정성 비교 추가
- [x] 양자 인코딩 5가지(basic, angle, iqp, dense_angle, amplitude)와 차원 축소 2가지(pca, select) 추가
- [ ] 기존 변수 선택 절차(t-test, 로지스틱 기준)를 확인해 `ttest_logit`에 정확히 반영
- [ ] 데이터를 넣고 노트북 실행 (먼저 작은 조합으로 오류 확인. seed 42, fixed16, basic/pca의 n = 8, 16이 이전 결과와 비슷한지 확인)
- [ ] fixed16과 훈련셋 선택 방법의 성능 차이로 선택 편향 크기 확인
- [ ] `results/results.csv`로 큐비트 수별 성능 그래프(오차 막대 포함) 그리기

## 참고

노트북 실행 결과(출력)는 git 필터로 커밋할 때 지워서 올립니다. 로컬 파일의 출력은 그대로 남습니다. 필터 설정은 `.gitattributes`와 각자의 `.git/config`에 있습니다.
