"""
공통 단계: 데이터 읽기 · EDA · 전처리 · 16개 변수 추출 · 분할

세 노트북(ind16, ind16_QSVM, ind16_QSVM_KTA)에 똑같이 들어 있던 셀을
한 곳으로 옮겼다. 계산 로직은 원래 셀과 같다.
"""

from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.utils.validation import check_is_fitted

from config import (
    DATA_PATH,
    SAVE_RESULTS,
    OUTPUT_DIR,
    CORR_THRESHOLD,
    SEED,
    ID, INDUSTRY, TARGET, DATE, AUDIT,
    ID_COL, TARGET_COL, DATE_COL,
    CATEGORICAL_COLS,
    SUSPECTED_LEAKAGE_COLS,
    MISSING_THRESHOLD,
    LOWER_QUANTILE,
    UPPER_QUANTILE,
    FEATURES_16,
    FEATURE_SELECTION,
    N_FEATURES,
    SELECTION_EXCLUDE,
    TTEST_ALPHA,
)


pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 180)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")


# ============================================================
# 2. PDF 참고: 산업코드 사전
# ============================================================

INDUSTRIES = {
    "D21": "펄프, 종이 및 종이제품 제조업",
    "D23": "코크스, 석유정제품 및 핵연료 제조업",
    "D24": "화합물 및 화학제품 제조업",
    "D25": "고무 및 플라스틱제품 제조업",
    "D26": "비금속 광물제품 제조업",
    "D27": "제1차 금속산업",
    "D28": "조립금속제품 제조업: 기계 및 가구 제외",
    "D29": "기타 기계 및 장비 제조업",
    "D30": "컴퓨터 및 사무용 기기제조업",
    "D31": "기타 전기기계 및 전기변환장치 제조업",
    "D32": "전자부품, 영상, 음향 및 통신장비 제조업",
    "D33": "의료, 정밀, 공학기기 및 시계 제조업",
    "D34": "자동차 및 트레일러 제조업",
    "D35": "기타 운송장비 제조업",
}


# ============================================================
# 3. PDF 참고: 변수 분류
#
# 변수명으로 직접 매칭한다.
# 비율, 금액, 배수, 기간 등이 섞여 있으므로
# 모든 수치 변수를 같은 단위라고 가정하면 안 된다.
# ============================================================

GROUP_NAMES = {
    "성장성": """
        매출액증가율|영업이익증가율|순이익증가율|총자산증가율|
        자기자본증가율|법인세차감전순이익증가율|유동자산증가율|
        유형자산증가율|재고자산증가율|종업원수증가율
    """,

    "수익성": """
        매출원가율|판관비율|영업이익률|법인세차감전순이익률|
        당기순이익율|ROIC|ROE|EBITDA마진율|재료비/매출액|
        노무비/매출액|경비/매출액|총자산법인세차감전순이익률|
        순금융비용부담율|자본금법인세차감전순이익률|감가상각률|
        감가상각비대총비용비율|법인세차감전순이익이자보상율|
        금융비용대총부채비율|금융비용대총비용비율|매출총이익률|
        배당성향|배당율|사내유보대자기자본비율|사내유보율|
        순외환손익대매출액|연구개발비대매출액|인건비대총비용비율|
        자기자본법인세차감전순이익률|자기자본배당률|자본금순이익률|
        조세공과대총비용비율|조세대조세차감전순이익률|차입금평균이자율|
        총자산순이익률(ROA)|총자산영업이익률|EBITDA(천원)|
        영업외수지율|경영자본영업이익율|수지비율|
        현금영업이익/매출액|현금흐름/매출액|FCF/매출액
    """,

    "안정성": """
        이자보상배수(배)|EBITDA/금융비용(배)|EBITDA/총차입금|
        차입금의존도|단기성차입금/총차입금|고정장기적합율|
        금융비용부담율|부채비율|자기자본비율|차입금/매출액|
        유동비율|EBITDA/단기차입금|차입금/자기자본비율|
        장단기대여금/총자산|적립금비율|비유동부채대순운전자본비율|
        비유동부채비율|비유동비율|당좌비율|매입채무대재고자산비율|
        매출채권대매입채무비율|매출채권대상/제품비율|법인세부담률|
        순운전자본대총자본비율|유동부채비율|유동자산구성비율|
        유보액대납입자본배율|유보액대총자산비율|
        재고자산대순운전자본비율|재고자산대유동자산비율|
        재무레버리지|차입금대월매출액|총현금흐름대당기상환부채|
        총현금흐름대매출액비율|총현금흐름대부채비율|
        총현금흐름대유동부채|총현금흐름대차입금비율|
        총현금흐름대총자본비율|비유동자산비율|단기부채상환능력|
        부채상환계수|총자산부채비율|적립금총자산비율|
        적립금비율2|현금비율|현금영업이익/금융비용|
        현금영업이익/단기차입금|현금영업이익/총차입금|
        현금흐름/총차입금|영업활동조달현금/금융비용|
        영업활동조달현금/차입금|FCF/차입금
    """,

    "활동성": """
        매출채권회전율(회)|재고자산회전율(회)|순운전자본회전율(회)|
        총자본회전율(회)|회전기간|비유동자산회전율(회)|
        매입채무회전기간|매입채무회전율(회)|매출채권회전기간|
        상(제)품회전율(회)|순영업자본회전율(회)|영업자본회전율(회)|
        순운전자본회전기간|원부재료회전율(회)|유동자산회전율(회)|
        유형자산회전율(회)|자기자본회전율(회)|자본금회전율(회)|
        재고자산회전기간|재공품회전율(회)|투하자산회전율(회)
    """,

    "생산성": """
        기계투자효율|노동소득분배율|부가가치(천원)|부가가치율|
        설비투자효율|이윤분배율|종업원1인당법인세차감전순이익|
        종업원1인당기계장비율|종업원1인당노동장비율|
        종업원1인당매출액|종업원1인당부가가치|종업원1인당순이익|
        종업원1인당영업이익|종업원1인당인건비|
        종업원1인당자본집약도|총자본투자효율|자본분배율
    """,

    "비재무/규모": """
        업력|경영자본|부가가치|종업원수|감사의견코드
    """,
}

GROUPS = {
    variable.strip(): group
    for group, variables in GROUP_NAMES.items()
    for variable in variables.split("|")
    if variable.strip()
}


# ============================================================
# 4. 데이터 읽기
# ============================================================

def read_data(path):
    """
    실제 파일 형식에 따라 xls/xlsx 엔진을 선택한다.

    회사코드는 식별자이므로 문자열로 읽는다.
    원본 파일 자체는 수정하지 않는다.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"파일이 없습니다: {path.resolve()}")

    with path.open("rb") as f:
        signature = f.read(4)

    engine = "openpyxl" if signature == b"PK\x03\x04" else "xlrd"

    df = pd.read_excel(
        path,
        sheet_name=0,
        engine=engine,
        dtype={
            ID: "string",
            INDUSTRY: "string",
            AUDIT: "string",
        },
    )

    df.columns = df.columns.astype(str).str.strip()

    if df.columns.duplicated().any():
        raise ValueError("중복된 열 이름이 있습니다.")

    required = {ID, INDUSTRY, TARGET, DATE, AUDIT}
    missing_columns = required - set(df.columns)

    if missing_columns:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")

    # 문자열의 앞뒤 공백만 정리
    for col in [ID, INDUSTRY, AUDIT]:
        df[col] = df[col].str.strip().replace("", pd.NA)

    return df


# ============================================================
# 5. 전체 EDA
# ============================================================

def explore_data(df, corr_threshold=0.95):
    """
    원자료를 임의 대치하거나 이상값을 삭제하지 않는 EDA.

    반환값
    -------
    x       : 수치형 분석용 복사본
    dates   : 파싱한 결산일
    tables  : 탐색 결과 DataFrame 사전
    summary : 전체 요약 사전
    """

    if not 0 <= corr_threshold <= 1:
        raise ValueError("상관계수 기준은 0~1 사이여야 합니다.")

    excluded = {ID, INDUSTRY, TARGET, DATE, AUDIT}

    numeric_features = [
        col for col in df.columns
        if col not in excluded
    ]

    # 숫자 변환 실패를 별도로 집계
    converted = df[numeric_features].apply(
        pd.to_numeric,
        errors="coerce",
    )

    conversion_failures = (
        df[numeric_features].notna()
        & converted.isna()
    )

    infinite = pd.DataFrame(
        np.isinf(converted),
        index=df.index,
        columns=numeric_features,
    )

    # 분석 복사본에서만 무한대를 NaN으로 처리
    # 원본의 결측, 0, 음수, 큰 수는 임의 변경하지 않음
    x = converted.replace([np.inf, -np.inf], np.nan)

    # --------------------------------------------------------
    # 결산일
    # --------------------------------------------------------

    date_text = (
        df[DATE]
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )

    dates = pd.to_datetime(
        date_text,
        format="%Y%m%d",
        errors="coerce",
    )

    year = dates.dt.year.astype("Int64").rename("결산연도")

    # 타깃 결측이 있더라도 교차표에서 제외되지 않게 표시
    labels = df[TARGET].astype("string").fillna("(결측)")
    industries = df[INDUSTRY].fillna("(결측)")
    audits = df[AUDIT].fillna("(결측)")

    # --------------------------------------------------------
    # A. 변수 프로파일
    # --------------------------------------------------------

    profile = pd.DataFrame({
        "변수": df.columns,
        "자료형": df.dtypes.astype(str).values,
        "분류": [
            GROUPS.get(col, "식별/분류/기준일 또는 미분류")
            for col in df.columns
        ],
        "결측수": df.isna().sum().values,
        "결측비율": df.isna().mean().values,
        "고유값수": df.nunique(dropna=True).values,
    })

    # --------------------------------------------------------
    # B. 수치형 기술통계
    # --------------------------------------------------------

    stats = x.describe(
        percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    ).T

    stats.index.name = "변수"

    valid_count = x.notna().sum().replace(0, np.nan)

    stats["결측비율_분석용"] = x.isna().mean()
    stats["0개수"] = x.eq(0).sum()
    stats["0비율_전체행기준"] = x.eq(0).mean()
    stats["0비율_유효값기준"] = x.eq(0).sum() / valid_count
    stats["음수개수"] = x.lt(0).sum()
    stats["음수비율_유효값기준"] = x.lt(0).sum() / valid_count
    stats["무한대개수"] = infinite.sum()
    stats["숫자변환실패수"] = conversion_failures.sum()
    stats["왜도"] = x.skew()
    stats["첨도"] = x.kurt()

    # --------------------------------------------------------
    # C. IQR 기반 극단값 탐색
    # --------------------------------------------------------

    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outlier_flags = (
        x.lt(lower)
        | x.gt(upper)
    ) & x.notna()

    stats["IQR하한"] = lower
    stats["IQR상한"] = upper
    stats["IQR극단값수"] = outlier_flags.sum()
    stats["IQR극단값비율"] = outlier_flags.sum() / valid_count
    stats["IQR이0인지"] = iqr.eq(0)

    # 주의:
    # IQR 기준을 벗어났다는 이유만으로 오류라고 판단하지 않음.
    # 분모가 작은 재무비율은 정상적인 값도 매우 커질 수 있음.
    # IQR=0인 변수에서는 특히 기계적인 해석을 피할 것.

    # --------------------------------------------------------
    # D. 완전히 동일한 변수
    # --------------------------------------------------------

    identical_pairs = []

    for i, col1 in enumerate(numeric_features):
        for col2 in numeric_features[i + 1:]:
            if df[col1].equals(df[col2]):
                identical_pairs.append((col1, col2))

    identical_table = pd.DataFrame(
        identical_pairs,
        columns=["변수1", "변수2"],
    )

    # --------------------------------------------------------
    # E. 상태 코드 빈도
    # --------------------------------------------------------

    target_counts = (
        labels
        .value_counts(dropna=False)
        .rename_axis(TARGET)
        .reset_index(name="행수")
    )

    target_counts["비율"] = target_counts["행수"] / len(df)

    # --------------------------------------------------------
    # F. 산업/연도/감사의견과 상태 코드
    # --------------------------------------------------------

    industry_target = pd.crosstab(
        industries,
        labels,
    )

    industry_target.index.name = INDUSTRY
    industry_target.columns.name = TARGET

    industry_target.insert(
        0,
        "산업명",
        [
            INDUSTRIES.get(code, "미확인")
            for code in industry_target.index
        ],
    )

    # -1은 결산일 파싱 실패 또는 결측
    year_target = pd.crosstab(
        year.fillna(-1),
        labels,
    )

    year_target.index.name = "결산연도"

    # 각 상태 코드 내부에서 연도가 차지하는 비율
    year_share_within_target = year_target.div(
        year_target.sum(axis=0).replace(0, np.nan),
        axis=1,
    )

    # 각 연도 내부에서 상태 코드가 차지하는 비율
    target_share_within_year = year_target.div(
        year_target.sum(axis=1).replace(0, np.nan),
        axis=0,
    )

    audit_target = pd.crosstab(
        audits,
        labels,
    )

    audit_target.index.name = AUDIT

    # --------------------------------------------------------
    # G. 상태 코드별 수치 통계와 결측률
    # --------------------------------------------------------

    grouped_stats = []
    grouped_missing = []

    for label in labels.unique():
        mask = labels.eq(label)
        block = x.loc[mask]

        part = (
            block
            .agg(["count", "mean", "median", "std"])
            .T
            .rename_axis("변수")
            .reset_index()
        )

        part.insert(0, TARGET, label)
        grouped_stats.append(part)

        missing_part = (
            block.isna()
            .mean()
            .rename_axis("변수")
            .reset_index(name="결측비율")
        )

        missing_part.insert(0, TARGET, label)
        grouped_missing.append(missing_part)

    stats_by_target = pd.concat(
        grouped_stats,
        ignore_index=True,
    )

    missing_by_target = pd.concat(
        grouped_missing,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # H. 상관관계
    # --------------------------------------------------------

    correlation_tables = {}

    for method in ["pearson", "spearman"]:
        corr = x.corr(
            method=method,
            min_periods=30,
        )

        pairs = []

        for i, col1 in enumerate(numeric_features):
            for col2 in numeric_features[i + 1:]:
                value = corr.loc[col1, col2]

                if pd.notna(value) and abs(value) >= corr_threshold:
                    n_pairs = (
                        x[col1].notna()
                        & x[col2].notna()
                    ).sum()

                    pairs.append({
                        "변수1": col1,
                        "변수2": col2,
                        "상관계수": value,
                        "유효쌍수": int(n_pairs),
                    })

        high_corr = pd.DataFrame(
            pairs,
            columns=["변수1", "변수2", "상관계수", "유효쌍수"],
        )

        if not high_corr.empty:
            high_corr = high_corr.sort_values(
                "상관계수",
                key=lambda s: s.abs(),
                ascending=False,
            )

        correlation_tables[f"{method}_matrix"] = corr
        correlation_tables[f"{method}_highcorr"] = high_corr

    # --------------------------------------------------------
    # I. 기업별 행 수
    # --------------------------------------------------------

    company_counts = (
        df[ID]
        .value_counts(dropna=False)
        .rename_axis(ID)
        .reset_index(name="행수")
    )

    # --------------------------------------------------------
    # J. 산업 × 연도 × 상태
    # --------------------------------------------------------

    cross = pd.DataFrame({
        "결산연도": year,
        INDUSTRY: industries,
        TARGET: labels,
    })

    year_industry_target = (
        cross
        .groupby(
            ["결산연도", INDUSTRY, TARGET],
            dropna=False,
        )
        .size()
        .reset_index(name="행수")
    )

    # --------------------------------------------------------
    # K. 파싱 이상 행
    # --------------------------------------------------------

    parse_issues = df[[ID, DATE]].copy()

    # 첫 행이 헤더라고 가정한 원본 Excel 행 번호
    parse_issues.insert(
        0,
        "엑셀행",
        np.arange(len(df)) + 2,
    )

    parse_issues["결산일파싱실패"] = dates.isna()
    parse_issues["숫자변환실패수"] = conversion_failures.sum(axis=1)
    parse_issues["무한대수"] = infinite.sum(axis=1)

    issue_mask = (
        parse_issues["결산일파싱실패"]
        | parse_issues["숫자변환실패수"].gt(0)
        | parse_issues["무한대수"].gt(0)
    )

    parse_issues = parse_issues.loc[issue_mask]

    # --------------------------------------------------------
    # L. 변수 사전
    # --------------------------------------------------------

    variable_dictionary = profile[["변수", "분류"]].copy()

    cautions = {
        TARGET: (
            "1/2의 의미와 상태 판정시점 확인 필요. "
            "현재 상태 분류와 미래 부도 예측을 구분할 것."
        ),
        DATE: (
            "예측시점 및 재무정보 이용 가능 시점 확인 필요. "
            "상태별 결산연도 분포 차이를 점검할 것."
        ),
        ID: "기업 식별자. 일반적으로 예측 독립변수에서 제외.",
        "업력": (
            "PDF K1 산식의 기업도산일 참조에 따른 "
            "예측시점 이후 정보 포함 가능성 확인 필요."
        ),
        "순운전자본회전기간": (
            "명칭과 PDF 산식 간 불일치 가능성 확인 필요."
        ),
        AUDIT: (
            "숫자형 연속변수가 아닌 범주형 변수. "
            "미확인 코드의 의미를 임의로 지정하지 말 것."
        ),
    }

    variable_dictionary["주의사항"] = (
        variable_dictionary["변수"]
        .map(cautions)
        .fillna("")
    )

    # --------------------------------------------------------
    # M. 결과 사전
    # --------------------------------------------------------

    tables = {
        "profile": profile,
        "numeric_stats": stats.reset_index(),
        "target_counts": target_counts,
        "industry_target": industry_target.reset_index(),
        "year_target": year_target.reset_index(),
        "year_share_within_target": (
            year_share_within_target.reset_index()
        ),
        "target_share_within_year": (
            target_share_within_year.reset_index()
        ),
        "audit_target": audit_target.reset_index(),
        "identical_columns": identical_table,
        "stats_by_target": stats_by_target,
        "missing_by_target": missing_by_target,
        "company_counts": company_counts,
        "year_industry_target": year_industry_target,
        "parse_issues": parse_issues,
        "variable_dictionary": variable_dictionary,
        "duplicate_rows": df.loc[df.duplicated(keep=False)],
    }

    tables.update(correlation_tables)

    summary = {
        "행수": len(df),
        "열수": len(df.columns),
        "고유기업수": int(df[ID].nunique()),
        "수치분석변수수": len(numeric_features),
        "전체결측셀수": int(df.isna().sum().sum()),
        "완전중복행수": int(df.duplicated().sum()),
        "회사_결산일중복수": int(
            df.duplicated([ID, DATE]).sum()
        ),
        "결산일최소": str(dates.min()),
        "결산일최대": str(dates.max()),
        "결산일파싱실패수": int(dates.isna().sum()),
        "숫자변환실패셀수": int(
            conversion_failures.sum().sum()
        ),
        "무한대셀수": int(infinite.sum().sum()),
        "전체결측변수": x.columns[
            x.notna().sum().eq(0)
        ].tolist(),
        "상수변수_결측제외": x.columns[
            x.nunique(dropna=True).eq(1)
        ].tolist(),
        "완전동일변수쌍": identical_pairs,
        "산업사전미등록코드": sorted(
            set(df[INDUSTRY].dropna()) - set(INDUSTRIES)
        ),
    }

    return x, dates, tables, summary


# ============================================================
# 6. 보기 좋게 출력하기
# ============================================================

def show_table(title, table, n=20):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    shown = table if n is None else table.head(n)

    try:
        from IPython.display import display
        display(shown)
    except ImportError:
        print(shown.to_string(index=False))


# ============================================================
# 7. 선택 사항: 탐색 결과 저장
# ============================================================

def save_results(tables, summary, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 상관행렬은 변수명이 인덱스에 있으므로 함께 저장
    matrix_names = {"pearson_matrix", "spearman_matrix"}

    with pd.ExcelWriter(
        out / "eda_tables.xlsx",
        engine="openpyxl",
    ) as writer:

        for name, table in tables.items():
            include_index = name in matrix_names

            table.to_excel(
                writer,
                sheet_name=name[:31],
                index=include_index,
            )

            table.to_csv(
                out / f"{name}.csv",
                index=include_index,
                encoding="utf-8-sig",
            )

            ws = writer.sheets[name[:31]]
            ws.freeze_panes = "B2" if include_index else "A2"
            ws.auto_filter.ref = ws.dimensions

    (out / "summary.json").write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n결과 저장 위치: {out.resolve()}")


# ============================================================
# 원본 파일 읽기 (data/ 기준)
#
# 기존 노트북과 같이 pd.read_excel 로 그대로 읽는다.
# (read_data 는 정의만 되어 있고 기존에도 사용하지 않았다.)
# ============================================================

def load_raw(path=DATA_PATH):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"데이터 파일이 없습니다: {path}\n"
            "원본 파일을 프로젝트의 data/ 폴더에 넣어 주세요."
        )

    return pd.read_excel(path)


# ============================================================
# 전체 탐색 실행 (기존 '3. 전체 탐색 실행' 셀)
# ============================================================

def run_eda(df):
    # ============================================================
    # 데이터 로딩
    # ============================================================

    print("데이터 크기:", df.shape)

    show_table(
        "원본 데이터 앞부분",
        df,
        n=5,
    )


    # ============================================================
    # EDA 실행
    # ============================================================

    x, dates, tables, summary = explore_data(
        df,
        corr_threshold=CORR_THRESHOLD,
    )

    print("\n[전체 요약]")
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )


    # ============================================================
    # 1. 변수 구성
    # ============================================================

    show_table(
        "변수별 자료형 / 분류 / 결측 / 고유값",
        tables["profile"],
        n=None,
    )

    show_table(
        "변수 분류별 개수",
        tables["profile"]
        .groupby("분류", dropna=False)
        .size()
        .reset_index(name="변수수"),
    )


    # ============================================================
    # 2. 결측치
    # ============================================================

    show_table(
        "결측률 상위 변수",
        tables["profile"].sort_values(
            "결측비율",
            ascending=False,
        ),
        n=30,
    )

    # 행별 결측률도 확인
    row_missing_rate = x.isna().mean(axis=1)

    show_table(
        "기업별 수치변수 결측률 요약",
        row_missing_rate.describe(
            percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        )
        .rename_axis("통계량")
        .reset_index(name="결측비율"),
    )

    show_table(
        "결측률이 높은 기업",
        df[[ID, INDUSTRY, TARGET, DATE]]
        .assign(수치변수결측비율=row_missing_rate)
        .sort_values(
            "수치변수결측비율",
            ascending=False,
        ),
        n=20,
    )


    # ============================================================
    # 3. 수치 분포 / 0 / 음수 / 극단값
    # ============================================================

    show_table(
        "수치변수 기술통계",
        tables["numeric_stats"],
        n=20,
    )

    show_table(
        "0 비율이 높은 변수",
        tables["numeric_stats"].sort_values(
            "0비율_유효값기준",
            ascending=False,
        ),
        n=20,
    )

    show_table(
        "음수 비율이 높은 변수",
        tables["numeric_stats"].sort_values(
            "음수비율_유효값기준",
            ascending=False,
        ),
        n=20,
    )

    show_table(
        "IQR 기준 극단값 비율이 높은 변수 — 자동 삭제 대상은 아님",
        tables["numeric_stats"].sort_values(
            "IQR극단값비율",
            ascending=False,
        ),
        n=25,
    )


    # ============================================================
    # 4. 타깃 분포
    # ============================================================

    show_table(
        "정상/폐업 코드별 행 수",
        tables["target_counts"],
    )

    show_table(
        "산업별 상태 코드 분포",
        tables["industry_target"],
        n=None,
    )

    show_table(
        "감사의견별 상태 코드 분포",
        tables["audit_target"],
        n=None,
    )


    # ============================================================
    # 5. 시점 교란 점검
    # ============================================================

    show_table(
        "결산연도별 상태 코드 수",
        tables["year_target"],
        n=None,
    )

    show_table(
        "각 상태 코드 내부에서 연도별 구성비",
        tables["year_share_within_target"],
        n=None,
    )

    show_table(
        "각 연도 내부에서 상태 코드별 구성비",
        tables["target_share_within_year"],
        n=None,
    )


    # ============================================================
    # 6. 상태 코드별 결측 차이
    # ============================================================

    missing_comparison = (
        tables["missing_by_target"]
        .pivot(
            index="변수",
            columns=TARGET,
            values="결측비율",
        )
    )

    # 여러 상태 코드 사이의 결측률 최대 차이
    missing_comparison["결측률최대차이"] = (
        missing_comparison.max(axis=1)
        - missing_comparison.min(axis=1)
    )

    show_table(
        "상태 코드에 따라 결측률 차이가 큰 변수",
        missing_comparison
        .sort_values(
            "결측률최대차이",
            ascending=False,
        )
        .reset_index(),
        n=30,
    )


    # ============================================================
    # 7. 주요 재무변수의 코드별 통계
    # ============================================================

    KEY_VARIABLES = [
        "영업이익률",
        "총자산순이익률(ROA)",
        "ROE",
        "부채비율",
        "자기자본비율",
        "유동비율",
        "차입금의존도",
        "이자보상배수(배)",
        "현금흐름/매출액",
        "종업원수",
        "업력",
    ]

    show_table(
        "주요 변수의 상태 코드별 통계",
        tables["stats_by_target"]
        .loc[
            lambda t: t["변수"].isin(KEY_VARIABLES)
        ]
        .sort_values(["변수", TARGET]),
        n=None,
    )

    median_comparison = (
        tables["stats_by_target"]
        .pivot(
            index="변수",
            columns=TARGET,
            values="median",
        )
    )

    show_table(
        "주요 변수의 코드별 중앙값",
        median_comparison
        .reindex(
            [
                col for col in KEY_VARIABLES
                if col in median_comparison.index
            ]
        )
        .reset_index(),
        n=None,
    )


    # ============================================================
    # 8. 중복 변수 / 다중공선성 후보
    # ============================================================

    show_table(
        "완전히 동일한 변수 쌍",
        tables["identical_columns"],
        n=None,
    )

    show_table(
        "Pearson 고상관 변수 쌍",
        tables["pearson_highcorr"],
        n=30,
    )

    show_table(
        "Spearman 고상관 변수 쌍",
        tables["spearman_highcorr"],
        n=30,
    )


    # ============================================================
    # 9. 데이터 구조 / 파싱 오류
    # ============================================================

    show_table(
        "기업별 관측 행 수",
        tables["company_counts"].sort_values(
            "행수",
            ascending=False,
        ),
        n=10,
    )

    show_table(
        "파싱 이상 행",
        tables["parse_issues"],
        n=20,
    )

    show_table(
        "해석 시 주의사항이 있는 변수",
        tables["variable_dictionary"].loc[
            lambda t: t["주의사항"].ne("")
        ],
        n=None,
    )


    # ============================================================
    # 10. 선택 저장
    # ============================================================

    if SAVE_RESULTS:
        save_results(
            tables,
            summary,
            OUTPUT_DIR,
        )

    return x, dates, tables, summary


# ============================================================
# 2. 훈련셋에서 수치변수 선택
#
# - 결측률 과다 변수
# - 상수/전체 결측 변수
# - 완전히 같은 수치변수
#
# 해당 판단은 fit()에 전달된 데이터에서만 수행한다.
# ============================================================

class NumericColumnSelector(BaseEstimator, TransformerMixin):

    def __init__(self, missing_threshold=0.60):
        self.missing_threshold = missing_threshold

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("입력은 pandas DataFrame이어야 합니다.")

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)

        self.missing_rates_ = X.isna().mean()

        self.high_missing_cols_ = self.missing_rates_[
            self.missing_rates_ > self.missing_threshold
        ].index.tolist()

        self.constant_cols_ = X.columns[
            X.nunique(dropna=True) <= 1
        ].tolist()

        excluded = set(
            self.high_missing_cols_ + self.constant_cols_
        )

        candidates = [
            col for col in X.columns
            if col not in excluded
        ]

        self.duplicate_mapping_ = {}
        retained = []

        for col in candidates:
            duplicate_of = next(
                (
                    previous
                    for previous in retained
                    if X[col].equals(X[previous])
                ),
                None,
            )

            if duplicate_of is None:
                retained.append(col)
            else:
                self.duplicate_mapping_[col] = duplicate_of

        if not retained:
            raise ValueError("선택 후 남는 수치변수가 없습니다.")

        self.selected_columns_ = retained
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_columns_")

        missing = set(self.selected_columns_) - set(X.columns)
        if missing:
            raise ValueError(f"필수 수치변수 누락: {missing}")

        return X.loc[:, self.selected_columns_].copy()

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "selected_columns_")
        return np.asarray(self.selected_columns_, dtype=object)


# ============================================================
# 3. 훈련셋 분위수 기반 극단값 제한
#
# 결측 여부가 보존되도록 대치 전에 실행한다.
# 행은 삭제하지 않는다.
# ============================================================

class QuantileClipper(BaseEstimator, TransformerMixin):

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError("0 <= lower < upper <= 1 이어야 합니다.")

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)

        self.lower_bounds_ = X.quantile(self.lower)
        self.upper_bounds_ = X.quantile(self.upper)

        # 0이 많은 변수에서는 두 분위수가 같아질 수 있다.
        # 그때 전체 값을 같은 값으로 압축하지 않도록 제한을 생략한다.
        usable = self.upper_bounds_ > self.lower_bounds_

        self.lower_bounds_ = self.lower_bounds_.where(
            usable,
            -np.inf,
        )

        self.upper_bounds_ = self.upper_bounds_.where(
            usable,
            np.inf,
        )

        return self

    def transform(self, X):
        check_is_fitted(self, "lower_bounds_")

        return X.clip(
            lower=self.lower_bounds_,
            upper=self.upper_bounds_,
            axis=1,
        )

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_


# ============================================================
# 4. 수치 결측 여부 표시
#
# 훈련셋에서 결측이 없었어도 미래 데이터에는 생길 수 있으므로
# 선택된 수치변수 전체에 대해 표시 변수를 만든다.
# ============================================================

class MissingFlags(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        return X.isna().astype(np.float64).to_numpy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(
            [f"{col}__missing" for col in self.feature_names_in_],
            dtype=object,
        )


# ============================================================
# 5. 입력 형태 정리
#
# 이 단계는 통계량을 학습하지 않는다.
# 추후 새로운 기업 데이터를 넣을 때도 같은 함수를 사용한다.
# ============================================================

def prepare_features(data):
    required = set(CATEGORICAL_COLS)

    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"범주형 변수 누락: {missing}")

    drop_cols = [
        ID_COL,
        TARGET_COL,
        DATE_COL,
        *SUSPECTED_LEAKAGE_COLS,
    ]

    X = data.drop(
        columns=drop_cols,
        errors="ignore",
    ).copy()

    # 범주형: 의미 미확인 코드는 그대로 유지한다.
    # DU, NS 등의 의미를 임의 추정하거나 순서를 부여하지 않음.
    for col in CATEGORICAL_COLS:
        values = X[col].astype("string").str.strip()
        values = values.mask(values.eq(""), pd.NA)

        # sklearn에서 일관되게 다루도록 np.nan으로 통일
        X[col] = values.astype(object)
        X.loc[values.isna(), col] = np.nan

    numeric_cols = [
        col for col in X.columns
        if col not in CATEGORICAL_COLS
    ]

    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X[numeric_cols] = X[numeric_cols].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return X


# ============================================================
# 전처리 실행 (기존 '기존 전처리' 셀의 6~10단계)
# ============================================================

def preprocess(df, seed=SEED):
    # ============================================================
    # 6. 타깃과 행 정리
    # ============================================================

    data = df.copy()
    data.columns = data.columns.astype(str).str.strip()

    required_columns = {
        ID_COL,
        TARGET_COL,
        DATE_COL,
        *CATEGORICAL_COLS,
    }

    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"필수 컬럼 누락: {missing_columns}")

    # 라벨 결측 행은 지도학습에서 제외
    target_text = (
        data[TARGET_COL]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .replace("", pd.NA)
    )

    print("타깃 결측으로 제외하는 행 수:", target_text.isna().sum())

    data = data.loc[target_text.notna()].copy()
    target_text = target_text.loc[data.index]

    # 완전히 동일한 행만 제거
    duplicate_mask = data.duplicated()
    print("완전 중복으로 제외하는 행 수:", duplicate_mask.sum())

    data = data.loc[~duplicate_mask].copy()
    target_text = target_text.loc[data.index]

    # 기업이 여러 행이면 기업 단위 분할이 필요하므로
    # 같은 기업이 train/test에 섞이지 않도록 여기서는 중단한다.
    company_ids = data[ID_COL].astype("string").str.strip()

    if company_ids.isna().any() or company_ids.eq("").any():
        raise ValueError("회사코드 결측을 먼저 확인해 주세요.")

    if company_ids.duplicated().any():
        raise ValueError(
            "기업당 여러 행이 있습니다. "
            "현재 코드 대신 기업 단위 그룹 분할을 사용해야 합니다."
        )

    observed_labels = set(target_text.unique())

    if observed_labels != {"1", "2"}:
        raise ValueError(
            f"예상하지 못한 타깃 코드: {observed_labels}"
        )

    # 의미를 추정하지 않고 단순히 이진 코드로 변환
    # 0 = 원본 코드 1
    # 1 = 원본 코드 2
    # 양성 클래스 1이 '폐업'이라는 뜻은 아직 아니다.
    TARGET_MAPPING = {"1": 0, "2": 1}

    y = target_text.map(TARGET_MAPPING).astype("int64")
    X = prepare_features(data)

    # 행 인덱스 정렬
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    metadata = data[
        [ID_COL, DATE_COL]
    ].reset_index(drop=True)

    metadata["결산연도"] = pd.to_datetime(
        metadata[DATE_COL]
        .astype("string")
        .str.replace(r"\.0$", "", regex=True),
        format="%Y%m%d",
        errors="coerce",
    ).dt.year

    numeric_cols = [
        col for col in X.columns
        if col not in CATEGORICAL_COLS
    ]

    print("\n타깃 매핑:", TARGET_MAPPING)
    print("전처리 전 X:", X.shape)
    print("수치형 변수 수:", len(numeric_cols))
    print("범주형 변수:", CATEGORICAL_COLS)


    # ============================================================
    # 7. 훈련 / 검증 / 테스트 분리
    #
    # 60% / 20% / 20%
    # 이 분할은 동일 표본 내부의 비교용이다.
    # 미래 시점 일반화 검증을 보장하지는 않는다.
    # ============================================================

    all_indices = np.arange(len(X))

    train_idx, remaining_idx = train_test_split(
        all_indices,
        test_size=0.40,
        stratify=y,
        random_state=seed,
    )

    valid_idx, test_idx = train_test_split(
        remaining_idx,
        test_size=0.50,
        stratify=y.iloc[remaining_idx],
        random_state=seed,
    )

    X_train = X.iloc[train_idx].copy()
    X_valid = X.iloc[valid_idx].copy()
    X_test = X.iloc[test_idx].copy()

    y_train = y.iloc[train_idx].copy()
    y_valid = y.iloc[valid_idx].copy()
    y_test = y.iloc[test_idx].copy()

    meta_train = metadata.iloc[train_idx].copy()
    meta_valid = metadata.iloc[valid_idx].copy()
    meta_test = metadata.iloc[test_idx].copy()


    # ============================================================
    # 8. 전처리 파이프라인
    # ============================================================

    # 수치 값:
    # 극단값 제한 → 중앙값 대치 → RobustScaler
    numeric_values_pipeline = Pipeline([
        (
            "clip",
            QuantileClipper(
                lower=LOWER_QUANTILE,
                upper=UPPER_QUANTILE,
            ),
        ),
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            RobustScaler(),
        ),
    ])

    # 선택한 수치변수를 값과 결측 여부 두 갈래로 처리
    numeric_branches = ColumnTransformer(
        transformers=[
            (
                "values",
                numeric_values_pipeline,
                slice(None),
            ),
            (
                "missing",
                MissingFlags(),
                slice(None),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    numeric_pipeline = Pipeline([
        (
            "select",
            NumericColumnSelector(
                missing_threshold=MISSING_THRESHOLD,
            ),
        ),
        (
            "branches",
            numeric_branches,
        ),
    ])

    # 산업코드와 감사의견에는 임의의 대소 관계를 부여하지 않는다.
    # 결측은 별도 범주로 처리.
    # 훈련 중 보지 못한 새 범주는 오류 대신 모두 0으로 처리된다.
    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="__MISSING__",
                keep_empty_features=True,
            ),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                dtype=np.float64,
            ),
        ),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_cols,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_COLS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


    # ============================================================
    # 9. 훈련셋에만 fit
    # ============================================================

    X_train_array = preprocessor.fit_transform(X_train, y_train)

    # 검증/테스트에서는 fit하지 않음!
    X_valid_array = preprocessor.transform(X_valid)
    X_test_array = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    X_train_processed = pd.DataFrame(
        X_train_array,
        index=X_train.index,
        columns=feature_names,
    )

    X_valid_processed = pd.DataFrame(
        X_valid_array,
        index=X_valid.index,
        columns=feature_names,
    )

    X_test_processed = pd.DataFrame(
        X_test_array,
        index=X_test.index,
        columns=feature_names,
    )


    # ============================================================
    # 10. 결과 검증
    # ============================================================

    for name, processed in [
        ("train", X_train_processed),
        ("valid", X_valid_processed),
        ("test", X_test_processed),
    ]:
        assert np.isfinite(processed.to_numpy()).all(), (
            f"{name}에 결측 또는 무한대가 남았습니다."
        )

        assert processed.columns.equals(
            X_train_processed.columns
        ), f"{name}의 열 구성이 다릅니다."

    assert set(meta_train[ID_COL]).isdisjoint(
        set(meta_valid[ID_COL])
    )

    assert set(meta_train[ID_COL]).isdisjoint(
        set(meta_test[ID_COL])
    )

    assert set(meta_valid[ID_COL]).isdisjoint(
        set(meta_test[ID_COL])
    )

    selector = (
        preprocessor
        .named_transformers_["numeric"]
        .named_steps["select"]
    )

    print("\n[전처리 완료]")
    print("훈련:", X_train_processed.shape)
    print("검증:", X_valid_processed.shape)
    print("테스트:", X_test_processed.shape)

    print("\n결측률 과다로 제거:")
    print(selector.high_missing_cols_)

    print("\n상수 또는 전체 결측으로 제거:")
    print(selector.constant_cols_)

    print("\n중복 제거 변수 → 남긴 변수:")
    print(selector.duplicate_mapping_)

    print("\n남긴 수치변수 수:")
    print(len(selector.selected_columns_))

    print("\n최종 독립변수 수:")
    print(len(feature_names))

    print("\n훈련 타깃 비율:")
    print(y_train.value_counts(normalize=True).sort_index())

    print("\n전처리 결과 앞부분:")
    print(X_train_processed.iloc[:5, :12])

    return SimpleNamespace(
        data=data,
        X=X,
        y=y,
        metadata=metadata,
        numeric_cols=numeric_cols,
        TARGET_MAPPING=TARGET_MAPPING,
        train_idx=train_idx,
        valid_idx=valid_idx,
        test_idx=test_idx,
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        meta_train=meta_train,
        meta_valid=meta_valid,
        meta_test=meta_test,
        preprocessor=preprocessor,
        feature_names=feature_names,
        X_train_processed=X_train_processed,
        X_valid_processed=X_valid_processed,
        X_test_processed=X_test_processed,
    )


# ============================================================
# 독립변수 16개 추출 (기존 '독립변수 16개 추출' 셀)
# 148개 전처리 결과에서 16개 변수만 뽑아 df16 만들기
# ============================================================

def find_col(processed, name):
    hits = [c for c in processed.columns if c.endswith("values__" + name)]
    assert len(hits) == 1, (name, hits)
    return hits[0]


def build_df16(prep):
    data = prep.data
    metadata = prep.metadata
    train_idx, valid_idx, test_idx = prep.train_idx, prep.valid_idx, prep.test_idx
    X_train_processed = prep.X_train_processed
    X_valid_processed = prep.X_valid_processed
    X_test_processed = prep.X_test_processed
    y_train, y_valid, y_test = prep.y_train, prep.y_valid, prep.y_test

    feat15 = [c for c in FEATURES_16 if c != '업력']   # 업력은 148개 전처리에서 빠져 있음

    # 1) 15개: 처리된 열 이름에서 값 열(values__)만 정확히 찾기

    col_map = {name: find_col(X_train_processed, name) for name in feat15}

    # 2) 업력(K1): 148개 버전에서 제외됐으므로 같은 방식으로 따로 처리 (훈련셋에만 fit)
    k1 = (
        pd.to_numeric(data["업력"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .reset_index(drop=True)
        .to_frame("업력")
    )

    k1_pipe = Pipeline([
        ("clip", QuantileClipper(lower=LOWER_QUANTILE, upper=UPPER_QUANTILE)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])
    k1_pipe.fit(k1.iloc[train_idx])

    def build16(processed, k1_part):
        out = processed[[col_map[c] for c in feat15]].copy()
        out.columns = feat15
        out["업력"] = k1_pipe.transform(k1_part).ravel()
        return out[FEATURES_16]

    X16_train = build16(X_train_processed, k1.iloc[train_idx])
    X16_valid = build16(X_valid_processed, k1.iloc[valid_idx])
    X16_test  = build16(X_test_processed,  k1.iloc[test_idx])

    # 3) df16: 전체를 하나로 (원래 행 순서, 타깃 + 식별/연도 포함)
    df16 = pd.concat([
        X16_train.assign(**{TARGET_COL: y_train}),
        X16_valid.assign(**{TARGET_COL: y_valid}),
        X16_test.assign(**{TARGET_COL: y_test}),
    ]).sort_index()

    df16.insert(0, ID_COL, metadata.loc[df16.index, ID_COL].values)
    df16["결산연도"] = metadata.loc[df16.index, "결산연도"].values

    print(df16.shape)        # (3140, 19)
    print("결측/무한대:", int(df16[FEATURES_16].isna().sum().sum()))

    return SimpleNamespace(
        df16=df16,
        feat15=feat15,
        col_map=col_map,
        k1_pipe=k1_pipe,
        X16_train=X16_train,
        X16_valid=X16_valid,
        X16_test=X16_test,
    )


# ============================================================
# df16 분할 (기존 '데이터 분할' 셀)
# ============================================================

def split16(df16, prep, seed=SEED):
    X = prep.X
    train_idx, valid_idx, test_idx = prep.train_idx, prep.valid_idx, prep.test_idx

    # ============================================================
    # df16 분할 (148개 버전과 동일한 분할 방식)
    # ============================================================
    # 148개 버전의 분할을 보관 (동일성 검증용)
    ref_train_idx, ref_valid_idx, ref_test_idx = (
        train_idx.copy(), valid_idx.copy(), test_idx.copy()
    )

    assert df16.index.equals(X.index), "df16 행 순서가 원본과 다릅니다."

    X16 = df16[FEATURES_16].copy()
    y16 = df16[TARGET_COL].astype(int).copy()
    meta16 = df16[[ID_COL, "결산연도"]].copy()
    industry16 = X.loc[df16.index, "산업코드"]     # 분포 확인용 (모델 입력 아님)

    assert len(X16) == len(y16) == len(meta16)
    assert set(y16.unique()) == {0, 1}

    indices = np.arange(len(X16))

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.40, random_state=seed,
        shuffle=True, stratify=y16,
    )

    valid_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=seed,
        shuffle=True, stratify=y16.iloc[temp_idx],
    )

    # 148개 버전과 행이 완전히 같은지 검증
    assert np.array_equal(train_idx, ref_train_idx)
    assert np.array_equal(valid_idx, ref_valid_idx)
    assert np.array_equal(test_idx, ref_test_idx)
    print("148개 버전과 동일한 분할 확인")

    X_train, X_valid, X_test = (X16.iloc[i].copy() for i in (train_idx, valid_idx, test_idx))
    y_train, y_valid, y_test = (y16.iloc[i].copy() for i in (train_idx, valid_idx, test_idx))
    meta_train, meta_valid, meta_test = (meta16.iloc[i].copy() for i in (train_idx, valid_idx, test_idx))

    assert set(train_idx).isdisjoint(valid_idx)
    assert set(train_idx).isdisjoint(test_idx)
    assert set(valid_idx).isdisjoint(test_idx)
    assert len(train_idx) + len(valid_idx) + len(test_idx) == len(X16)

    for left, right in [(meta_train, meta_valid), (meta_train, meta_test), (meta_valid, meta_test)]:
        assert set(left[ID_COL]).isdisjoint(set(right[ID_COL])), "같은 기업이 여러 세트에 포함됐습니다."

    split_summary = pd.DataFrame({
        "데이터": ["train", "valid", "test"],
        "행수": [len(y_train), len(y_valid), len(y_test)],
        "원본코드1_개수": [y_train.eq(0).sum(), y_valid.eq(0).sum(), y_test.eq(0).sum()],
        "원본코드2_개수": [y_train.eq(1).sum(), y_valid.eq(1).sum(), y_test.eq(1).sum()],
        "원본코드2_비율": [y_train.mean(), y_valid.mean(), y_test.mean()],
    })
    print("\n[분할 결과]")
    print(split_summary.to_string(index=False))

    industry_distribution = pd.DataFrame({
        "train": industry16.iloc[train_idx].value_counts(normalize=True),
        "valid": industry16.iloc[valid_idx].value_counts(normalize=True),
        "test":  industry16.iloc[test_idx].value_counts(normalize=True),
    }).fillna(0).sort_index()
    print("\n[산업별 구성비]")
    print(industry_distribution.round(3))

    year_distribution = pd.DataFrame({
        "train": meta_train["결산연도"].value_counts(normalize=True, dropna=False),
        "valid": meta_valid["결산연도"].value_counts(normalize=True, dropna=False),
        "test":  meta_test["결산연도"].value_counts(normalize=True, dropna=False),
    }).fillna(0).sort_index()
    print("\n[결산연도별 구성비]")
    print(year_distribution.round(3))

    return SimpleNamespace(
        X16=X16,
        y16=y16,
        meta16=meta16,
        train_idx=train_idx,
        valid_idx=valid_idx,
        test_idx=test_idx,
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        meta_train=meta_train,
        meta_valid=meta_valid,
        meta_test=meta_test,
        split_summary=split_summary,
        industry_distribution=industry_distribution,
        year_distribution=year_distribution,
    )


# ============================================================
# 변수 후보 (전처리 결과 148개 열 + 업력)
# ============================================================

def _pool_name(column):
    """전처리 열 이름을 읽기 쉬운 이름으로 (값 열은 원래 변수명)"""

    for prefix in ("numeric__values__", "numeric__missing__", "categorical__"):
        if column.startswith(prefix):
            return column[len(prefix):]

    return column


def build_pool(prep):
    """
    전체 행(원래 순서)의 변수 후보 표.
    - 전처리 결과 148개 열 (수치 값, 결측 표시, 범주 원핫) — 훈련셋에서만 fit 된 값
    - 업력 — build_df16 과 같은 방식으로 훈련셋에서만 fit
    """

    d16 = build_df16(prep)

    processed = pd.concat([
        prep.X_train_processed,
        prep.X_valid_processed,
        prep.X_test_processed,
    ]).sort_index()

    processed.columns = [_pool_name(col) for col in processed.columns]

    if processed.columns.duplicated().any():
        raise ValueError("변수 후보 이름이 중복됩니다.")

    pool = processed.copy()
    pool["업력"] = d16.df16.loc[pool.index, "업력"].to_numpy()

    # fixed16 이 기존 df16 과 같은 값인지 확인
    assert np.allclose(
        pool[FEATURES_16].to_numpy(),
        d16.df16.loc[pool.index, FEATURES_16].to_numpy(),
    ), "변수 후보의 16개 열이 기존 df16 과 다릅니다."

    return pool, d16


# ============================================================
# 변수 선택 (훈련셋에서만)
#
# 반환: 순위 순서의 변수 목록 (상위 k개), 전체 순위 표
# ============================================================

def _rank_by(scores, columns, tiebreak=None):
    table = pd.DataFrame({"변수": columns, "점수": np.asarray(scores, dtype=float)})

    if tiebreak is not None:
        table["보조"] = np.asarray(tiebreak, dtype=float)
        table = table.sort_values(["점수", "보조"], ascending=False)
    else:
        table = table.sort_values("점수", ascending=False)

    return table.reset_index(drop=True)


def select_features(
    method, X_train, y_train, k=N_FEATURES, seed=SEED, exclude=SELECTION_EXCLUDE,
):
    from scipy.stats import ttest_ind
    from sklearn.feature_selection import f_classif, mutual_info_classif
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y_train).reshape(-1)

    candidates = [col for col in X_train.columns if col not in set(exclude)]

    if method == "fixed16":
        chosen = [col for col in FEATURES_16 if col not in set(exclude)][:k]
        return chosen, pd.DataFrame({"변수": chosen, "점수": np.nan})

    # 훈련셋에서 상수인 열은 후보에서 뺀다
    X = X_train[candidates]
    X = X.loc[:, X.nunique() > 1]
    columns = list(X.columns)
    values = X.to_numpy(dtype=np.float64)

    if len(columns) < k:
        raise ValueError(f"후보 변수가 {len(columns)}개라 {k}개를 고를 수 없습니다.")

    # 동점 처리용 단변량 F-통계량
    f_score = np.nan_to_num(f_classif(values, y)[0])

    standardized = StandardScaler().fit_transform(values)

    if method == "ttest_logit":
        # 1) Welch t-test 로 두 집단 평균 차이가 유의한 변수
        p_values = ttest_ind(values[y == 1], values[y == 0], equal_var=False).pvalue
        p_values = np.nan_to_num(p_values, nan=1.0)

        passed = np.flatnonzero(p_values < TTEST_ALPHA)

        if len(passed) < k:
            # 유의한 변수가 k개보다 적으면 p값 순으로 채운다
            passed = np.argsort(p_values)[:k]

        # 2) 통과한 변수로 로지스틱 회귀 → 표준화 계수 크기 순
        logistic = LogisticRegression(max_iter=5000, random_state=seed)
        logistic.fit(standardized[:, passed], y)

        scores = np.zeros(len(columns))
        scores[passed] = np.abs(logistic.coef_[0]) + 1.0   # 통과 변수를 항상 앞에

        ranking = _rank_by(scores, columns, tiebreak=-p_values)
        ranking["p값"] = pd.Series(p_values, index=columns).loc[ranking["변수"]].to_numpy()

    elif method == "elasticnet":
        model = LogisticRegressionCV(
            Cs=10,
            cv=StratifiedKFold(5, shuffle=True, random_state=seed),
            penalty="elasticnet",
            solver="saga",
            l1_ratios=[0.5],
            scoring="roc_auc",
            max_iter=5000,
            random_state=seed,
            n_jobs=-1,
        ).fit(standardized, y)

        ranking = _rank_by(np.abs(model.coef_[0]), columns, tiebreak=f_score)

    elif method == "mutual_info":
        scores = mutual_info_classif(values, y, random_state=seed)
        ranking = _rank_by(scores, columns, tiebreak=f_score)

    elif method == "mrmr":
        # 관련성: F-통계량, 중복성: 이미 고른 변수와의 평균 |피어슨 상관|
        corr = np.nan_to_num(np.abs(np.corrcoef(standardized, rowvar=False)))

        selected = [int(np.argmax(f_score))]
        order_scores = [f_score[selected[0]]]
        remaining = set(range(len(columns))) - set(selected)

        while len(selected) < k:
            rest = np.array(sorted(remaining))
            redundancy = corr[np.ix_(rest, selected)].mean(axis=1)
            score = f_score[rest] / np.maximum(redundancy, 1e-6)

            best = int(rest[np.argmax(score)])
            selected.append(best)
            order_scores.append(float(score.max()))
            remaining.discard(best)

        ranking = pd.DataFrame({
            "변수": [columns[i] for i in selected],
            "점수": order_scores,
        })

    elif method == "xgb":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            importance_type="gain",
            random_state=seed,
            n_jobs=-1,
        ).fit(values, y)

        ranking = _rank_by(model.feature_importances_, columns, tiebreak=f_score)

    else:
        raise ValueError(
            f"알 수 없는 변수 선택 방법: {method}. "
            "사용 가능: fixed16, ttest_logit, elasticnet, mutual_info, mrmr, xgb"
        )

    return list(ranking["변수"].iloc[:k]), ranking


def selection_stability(selected_by_seed):
    """seed별 선택 결과의 평균 쌍별 Jaccard 지수 (1 = 매번 같은 변수)"""

    sets = [set(v) for v in selected_by_seed.values()]

    if len(sets) < 2:
        return np.nan

    scores = [
        len(a & b) / len(a | b)
        for i, a in enumerate(sets)
        for b in sets[i + 1:]
    ]

    return float(np.mean(scores))


# ============================================================
# seed 하나에 대한 데이터 준비 (여러 seed 반복 실험용)
#
# seed 마다 다시 하는 것:
#   train/valid/test 분할 → 전처리 fit → 변수 후보 → 변수 선택 (모두 훈련셋에서만)
# ============================================================

def prepare_split(df, seed=SEED, selection=FEATURE_SELECTION, k=N_FEATURES, verbose=False):
    """preprocess → build_pool → select_features 를 seed 하나로 실행"""

    import contextlib
    import io

    # 전처리 로그는 숨긴다 (verbose=True 이면 출력)
    sink = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())

    with sink:
        prep = preprocess(df, seed=seed)
        pool, d16 = build_pool(prep)

    index = (prep.train_idx, prep.valid_idx, prep.test_idx)
    Y = (prep.y_train, prep.y_valid, prep.y_test)

    selected, ranking = select_features(
        selection, pool.loc[index[0]], Y[0], k=k, seed=seed,
    )

    X = tuple(pool.loc[idx, selected] for idx in index)

    for part, target in zip(X, Y):
        assert part.index.equals(target.index), "X 와 y 의 행 순서가 다릅니다."

    print(
        f"[seed={seed}, 변수 선택={selection}] "
        f"train/valid/test = {len(Y[0])}/{len(Y[1])}/{len(Y[2])}, "
        f"원본코드2 비율 = {Y[0].mean():.3f}/{Y[1].mean():.3f}/{Y[2].mean():.3f}"
    )

    if verbose:
        print("선택된 변수 (순위 순):", selected)

    return SimpleNamespace(
        seed=seed,
        selection=selection,
        selected=selected,
        ranking=ranking,
        X=X,
        Y=Y,
        prep=prep,
        pool=pool,
        df16=d16.df16,
    )


def iterate_splits(df, selections, seeds, k=N_FEATURES, verbose=False):
    """(변수 선택 방법 × seed) 마다 prepare_split 결과를 차례로 돌려준다."""

    for selection in selections:
        for seed in seeds:
            yield prepare_split(df, seed=seed, selection=selection, k=k, verbose=verbose)
