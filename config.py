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

# 분류 임계값
THRESHOLD = 0.5


# ============================================================
# 양자 입력 인코딩
# ============================================================

# 8큐비트: 16개 변수 → PCA 8개 성분
N_QUBITS = 8

# 16큐비트: PCA 없이 변수 1개 = 큐비트 1개
N_QUBITS_16 = 16

# theta = pi * tanh(z / ANGLE_SCALE)
# 값이 클수록 tanh 포화가 완만해짐
# 테스트셋을 보고 조정하지 말고, 필요하면 검증셋에서 선택
ANGLE_SCALE = 2.0
ANGLE_SCALE_16 = 2.0


# ============================================================
# QSVM
# ============================================================

C_GRID = [0.1, 1, 10, 100, 1000]

# 특징 맵 반복 횟수 (QSVM 기본값)
QSVM_REPS = 2

# 8큐비트는 기본 설정으로 수 초 안에 끝난다.
GAMMAS_8Q = (0.03, 0.05, 0.1, 0.25)

# 16큐비트는 γ 하나당 수십 초 걸린다.
GAMMAS_16Q = (0.05, 0.1, 0.2, 0.3)

# 고전 기준선 RBF-SVM의 gamma 탐색 범위
CLASSICAL_GAMMAS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]


# ============================================================
# QSVM-KTA
# ============================================================

GAMMAS_KTA_8Q = (0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.25, 0.4)
GAMMAS_KTA_16Q = (0.01, 0.02, 0.03, 0.05, 0.1, 0.2)
REPS_KTA = (1, 2, 3)

# KTA 계산에 쓰는 훈련셋 부분표본 크기
N_KTA_8Q = 400
N_KTA_16Q = 300


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

# XGBoost + PCA 8개
N_COMPONENTS = 8

XGB_PCA_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=SEED,
    n_jobs=-1,
)


# ============================================================
# VQC
# ============================================================

N_LAYERS = 2
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 0.01

# VQC + data re-uploading: 데이터 인코딩 반복 횟수
N_REUPLOADS = 2
