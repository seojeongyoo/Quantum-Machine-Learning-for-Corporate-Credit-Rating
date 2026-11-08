"""
실험 설정 (세 노트북 공통)

경로는 이 파일의 위치(프로젝트 루트)를 기준으로 만든다.
데이터는 data/ 폴더에 두고, 결과는 results/results.csv 하나에 모은다.
"""

from pathlib import Path


# ============================================================
# 경로
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
DATA_FILE = "FILE2_중공업.xls"
DATA_PATH = DATA_DIR / DATA_FILE

RESULTS_DIR = ROOT / "results"
RESULTS_CSV = RESULTS_DIR / "results.csv"

# EDA 표 저장 여부와 위치 (기본: 저장하지 않음)
SAVE_RESULTS = False
OUTPUT_DIR = RESULTS_DIR / "eda"


# ============================================================
# 재현성
# ============================================================

SEED = 42

# 반복 실험에 쓸 seed 목록 (반복 무작위 분할 검증)
# seed 마다 train/valid/test 분할, 전처리 fit, 모델 난수가 달라지고
# 결과는 평균 ± 표준오차로 요약한다.
# 실행 시간은 seed 개수에 비례한다. 빠르게 확인할 때는 [SEED] 하나로 줄인다.
SEEDS = [42, 43, 44, 45, 46]

# 신뢰구간 수준
CONFIDENCE = 0.95


# ============================================================
# 컬럼
# ============================================================

ID = "회사코드"
INDUSTRY = "산업코드"
TARGET = "정상/폐업"
DATE = "재무비율기준/결산기준일"
AUDIT = "감사의견코드"

# 전처리 코드에서 쓰는 이름 (값은 위와 같다)
ID_COL = ID
TARGET_COL = TARGET
DATE_COL = DATE

CATEGORICAL_COLS = [
    "산업코드",
    "감사의견코드",
]

# 예측시점 이후 정보가 포함될 가능성이 있어 우선 제외
SUSPECTED_LEAKAGE_COLS = ["업력"]


# ============================================================
# EDA / 전처리
# ============================================================

# 고상관 변수 기준
CORR_THRESHOLD = 0.95

# 훈련셋에서 결측률이 60%를 초과하는 수치변수 제거
MISSING_THRESHOLD = 0.60

# 훈련셋 기준 양쪽 1% 극단값 제한
LOWER_QUANTILE = 0.01
UPPER_QUANTILE = 0.99

# 독립변수 16개 (t-test + logistic regression으로 도출)
FEATURES_16 = [
    '영업이익률', '재료비/매출액', '감가상각률', '금융비용대총부채비율',
    '금융비용대총비용비율', 'EBITDA/총차입금', '차입금의존도',
    '장단기대여금/총자산', '법인세부담률', '순운전자본대총자본비율',
    '유동자산구성비율', '유보액대총자산비율', '재고자산대유동자산비율',
    '매입채무회전기간', '자본금회전율(회)', '업력',
]

# ============================================================
# 변수 선택 (seed마다 그 seed의 훈련셋에서만 한다)
#
# 후보: 전처리 결과 148개 열 + 업력  →  순위를 매겨 상위 N_FEATURES 개
#
#   fixed16     : 위의 FEATURES_16 고정 (기존 방식. 전체 데이터로 고른 목록이라
#                 테스트 정보가 섞여 있다. 선택 편향 크기를 재는 비교용)
#   ttest_logit : Welch t-test (p < TTEST_ALPHA) 통과 변수 → 표준화 로지스틱 회귀 |계수| 순
#   elasticnet  : Elastic-net 로지스틱 회귀 (C 는 훈련셋 5-fold CV) |계수| 순
#   mutual_info : 상호정보량 순
#   mrmr        : mRMR (F-통계량 관련성 / 이미 고른 변수와의 평균 |상관| ) 탐욕 선택
#   xgb         : XGBoost gain 중요도 순
# ============================================================

FEATURE_SELECTION = "fixed16"

# 비교할 변수 선택 방법 목록 (노트북의 반복 실험이 이 목록을 모두 돈다)
SELECTION_METHODS = ["fixed16"]
# SELECTION_METHODS = ["fixed16", "ttest_logit", "elasticnet", "mutual_info", "mrmr", "xgb"]

# 모든 변수 선택 방법 (ind16 노트북의 XGBoost 비교는 항상 이 목록을 모두 돈다)
ALL_SELECTION_METHODS = ["fixed16", "ttest_logit", "elasticnet", "mutual_info", "mrmr", "xgb"]

# 고를 변수 수 (최대 큐비트 수와 같게)
N_FEATURES = 16

# 선택 후보에서 뺄 변수 (예: 누수 가능성이 있는 업력을 빼려면 ["업력"])
SELECTION_EXCLUDE = []

# ttest_logit 의 t-test 유의수준
TTEST_ALPHA = 0.05

# 분류 임계값
THRESHOLD = 0.5


# ============================================================
# 양자 입력 인코딩 (양자 모델에만 적용, 고전 모델은 변수 16개 그대로)
#
# 기본 인코딩 "basic":
#   n < 16 : 변수 16개 → PCA n개 성분 → θ = π·tanh(z / ANGLE_SCALE)
#   n = 16 : 변수 16개 그대로 (PCA 없음) → θ
#
# 다른 인코딩은 src/qsvm.py 의 ENCODERS 에 추가하고 여기서 이름을 바꾼다.
# ============================================================

ENCODING = "basic"

# 비교할 인코딩 목록 (노트북의 반복 실험이 이 목록을 모두 돈다)
#   basic       : 기존 방식 (QSVM = IQP 특징 맵, VQC = RY 각도)
#   angle       : 큐비트마다 RY(θ), 얽힘 없음
#   iqp         : IQP/ZZ 특징 맵 (QSVM, VQC 모두)
#   dense_angle : 큐비트 하나에 변수 2개 (RY, RZ) → n ≤ 8
#   amplitude   : 변수 2^n개를 상태 진폭으로 → n ≤ 4
# 쓸 수 없는 큐비트 수는 자동으로 건너뛴다.
ENCODINGS = ["basic"]
# ENCODINGS = ["basic", "angle", "iqp", "dense_angle", "amplitude"]

# 큐비트 수보다 입력 변수가 많을 때 줄이는 방법
#   "pca"    : PCA d개 성분 (기존 방식)
#   "select" : 변수 선택 순위 상위 d개
REDUCTION = "pca"
REDUCTIONS = ["pca"]
# REDUCTIONS = ["pca", "select"]

# 반복 실험할 큐비트 수 (1 ~ 16) — QSVM, QSVM-KTA
QUBIT_RANGE = list(range(1, 17))

# VQC / VQC+drc 큐비트 수 (1 ~ 8)
# PennyLane 시뮬레이션 학습은 큐비트마다 시간이 약 2배씩 늘어서
# 8큐비트(Colab 약 4.5분)를 넘으면 조합을 다 돌리기 어렵다.
VQC_QUBIT_RANGE = list(range(1, 9))

# 값이 클수록 tanh 포화가 완만해짐
# 테스트셋을 보고 조정하지 말고, 필요하면 검증셋에서 선택
ANGLE_SCALE = 2.0


# ============================================================
# QSVM
# ============================================================

# 기존 범위(0.1 ~ 1000)에 양 끝을 하나씩 추가 (작은 큐비트에서 끝 값이 선택됨)
C_GRID = [0.01, 0.1, 1, 10, 100, 1000, 10000]

# 특징 맵 반복 횟수 (QSVM 기본값)
QSVM_REPS = 2

# 각도 스케일 gamma 탐색 범위 (모든 큐비트 수 공통)
# 기존 8큐비트(0.03~0.25)와 16큐비트(0.05~0.3) 범위를 합치고,
# 큐비트가 적을 때를 위해 큰 값을 추가했다.
# 16큐비트는 γ 하나당 수십 초 걸린다.
QSVM_GAMMAS = (0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)

# 고전 기준선 RBF-SVM의 gamma 탐색 범위
CLASSICAL_GAMMAS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]


# ============================================================
# QSVM-KTA
# ============================================================

# 기존 8큐비트 / 16큐비트 범위를 합치고 큰 값을 추가
KTA_GAMMAS = (0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.4, 0.7, 1.0)
REPS_KTA = (1, 2, 3)

# KTA 계산에 쓰는 훈련셋 부분표본 크기
# (기존 16큐비트 값. 커널 행렬이 300 x 300)
N_KTA = 300


# ============================================================
# XGBoost
# ============================================================

XGB_PARAMS = dict(
    objective="binary:logistic",
    # 조기 종료가 있으므로 상한을 넉넉하게 설정
    n_estimators=2000,
    learning_rate=0.03,
    # 트리 복잡도 제한
    max_depth=3,
    min_child_weight=5,
    # 행 / 변수 샘플링
    subsample=0.8,
    colsample_bytree=0.8,
    # 규제
    reg_alpha=0.1,
    reg_lambda=5.0,
    # 기본 모델에서는 가중치 미조정
    scale_pos_weight=1.0,
    tree_method="hist",
    eval_metric="logloss",
    early_stopping_rounds=100,
    random_state=SEED,
    n_jobs=-1,
    importance_type="gain",
)


# ============================================================
# 고전 모델 (src/classical.py) — 선택된 변수 16개 그대로, PCA 없음
# 하이퍼파라미터는 검증셋 ROC-AUC로 선택
# ============================================================

# XGBoost: XGB_PARAMS 를 기본으로, 아래 조합을 덮어써 가며 탐색 (트리 개수는 조기 종료)
# 기존 노트북 설정(max_depth 3, learning_rate 0.03, min_child_weight 5, reg_lambda 5)이 포함돼 있다.
XGB_GRID = dict(
    max_depth=[3, 4, 6, 8, 10],
    learning_rate=[0.01, 0.03, 0.1],
    min_child_weight=[1, 5, 10],
    reg_lambda=[1.0, 5.0, 20.0],
)

RF_GRID = dict(
    n_estimators=[500],
    max_depth=[None, 4, 8, 12],
    min_samples_leaf=[1, 3, 5, 10],
    max_features=["sqrt", 0.5],
)

SVM_GRID = dict(
    C=[0.1, 0.3, 1, 3, 10, 30, 100],
    gamma=["scale", 0.003, 0.01, 0.03, 0.1, 0.3],
)

LOGISTIC_GRID = dict(
    C=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
    penalty=["l2", "l1"],
)


# ============================================================
# VQC (입력은 QSVM과 같은 인코딩, 큐비트 수는 VQC_QUBIT_RANGE)
# ============================================================

N_LAYERS = 2
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 0.01

# VQC + data re-uploading: 데이터 인코딩 반복 횟수
N_REUPLOADS = 2
