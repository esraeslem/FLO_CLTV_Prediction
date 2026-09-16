"""
FLO CLTV Prediction - BG-NBD & Gamma-Gamma ile Musteri Yasam Boyu Degeri
==========================================================================

Is Problemi
-----------
FLO, satis ve pazarlama faaliyetleri icin bir yol haritasi belirlemek istiyor.
Orta-uzun vadeli plan yapabilmek icin mevcut musterilerin gelecekte sirkete
saglayacagi potansiyel degerin (CLTV) tahmin edilmesi gerekiyor.

Yontem
------
- BG/NBD (Beta Geometric / Negative Binomial Distribution): musterinin
  gelecekte kac islem yapacagini (expected number of transactions) tahmin eder.
- Gamma-Gamma: musteri basina ortalama islem degerini (expected average
  profit) tahmin eder.
- Ikisi birlikte 6 aylik CLTV'yi hesaplar.

Kaynak: Miuul Data Science & Machine Learning Bootcamp - FLO CLTV Prediction

Kullanim
--------
    python src/flo_cltv_prediction.py
"""

import os

import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: "%.4f" % x)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "flo_data_20k.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


###############################################################
# GOREV 1: Veriyi Hazirlama
###############################################################

def outlier_thresholds(dataframe: pd.DataFrame, variable: str):
    """IQR (yuzde 1 - yuzde 99) tabanli alt/ust aykiri deger sinirlarini
    hesaplar. CLTV hesabinda frequency'nin integer olmasi gerektiginden
    sinirlar round() ile tam sayiya yuvarlanir (Adim 2, not).
    """
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = round(quartile3 + 1.5 * interquantile_range)
    low_limit = round(quartile1 - 1.5 * interquantile_range)
    return low_limit, up_limit


def replace_with_thresholds(dataframe: pd.DataFrame, variable: str) -> None:
    """Bir degiskendeki aykiri degerleri hesaplanan alt/ust sinirlarla baskilar (yerinde/in-place)."""
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


OUTLIER_COLUMNS = [
    "order_num_total_ever_online",
    "order_num_total_ever_offline",
    "customer_value_total_ever_offline",
    "customer_value_total_ever_online",
]


def data_prep(csv_path: str) -> pd.DataFrame:
    """Adim 1-5: veri okuma, kopya olusturma, aykiri deger baskilama,
    omnichannel toplam kolonlari ve tarih tipi donusumu.
    """
    dataframe = pd.read_csv(csv_path)
    dataframe = dataframe.copy()

    for col in OUTLIER_COLUMNS:
        replace_with_thresholds(dataframe, col)

    dataframe["order_num_total"] = (
        dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    )
    dataframe["customer_value_total"] = (
        dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    )

    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)

    return dataframe


###############################################################
# GOREV 2: CLTV Veri Yapisinin Olusturulmasi
###############################################################

def create_cltv_df(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Analiz tarihini son alisveris tarihinden 2 gun sonrasi alip BG/NBD ve
    Gamma-Gamma modellerinin bekledigi formatta (haftalik recency/tenure,
    islem basina ortalama harcama) bir cltv dataframe'i olusturur.
    """
    analysis_date = dataframe["last_order_date"].max() + pd.Timedelta(days=2)

    cltv_df = pd.DataFrame()
    cltv_df["customer_id"] = dataframe["master_id"]
    cltv_df["recency_cltv_weekly"] = (
        dataframe["last_order_date"] - dataframe["first_order_date"]
    ).dt.days / 7
    cltv_df["T_weekly"] = (analysis_date - dataframe["first_order_date"]).dt.days / 7
    cltv_df["frequency"] = dataframe["order_num_total"]
    cltv_df["monetary_cltv_avg"] = dataframe["customer_value_total"] / dataframe["order_num_total"]

    # BG/NBD ve Gamma-Gamma en az 2 alisveris yapmis musterileri bekler
    cltv_df = cltv_df[cltv_df["frequency"] > 1].reset_index(drop=True)

    return cltv_df


###############################################################
# GOREV 3: BG/NBD, Gamma-Gamma Modelleri ve CLTV Hesabi
###############################################################

def fit_bgnbd_model(cltv_df: pd.DataFrame, penalizer_coef: float = 0.001) -> BetaGeoFitter:
    """Adim 1: BG/NBD modelini fit eder; expected sales tahminlerini 3 ve
    6 ay icin cltv_df'e ekler.
    """
    bgf = BetaGeoFitter(penalizer_coef=penalizer_coef)
    bgf.fit(cltv_df["frequency"], cltv_df["recency_cltv_weekly"], cltv_df["T_weekly"])

    cltv_df["exp_sales_3_month"] = bgf.predict(
        4 * 3, cltv_df["frequency"], cltv_df["recency_cltv_weekly"], cltv_df["T_weekly"]
    )
    cltv_df["exp_sales_6_month"] = bgf.predict(
        4 * 6, cltv_df["frequency"], cltv_df["recency_cltv_weekly"], cltv_df["T_weekly"]
    )
    return bgf


def fit_gamma_gamma_model(cltv_df: pd.DataFrame, penalizer_coef: float = 0.01) -> GammaGammaFitter:
    """Adim 2: Gamma-Gamma modelini fit eder; musteri basina beklenen
    ortalama harcamayi (exp_average_value) cltv_df'e ekler.
    """
    ggf = GammaGammaFitter(penalizer_coef=penalizer_coef)
    ggf.fit(cltv_df["frequency"], cltv_df["monetary_cltv_avg"])

    cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(
        cltv_df["frequency"], cltv_df["monetary_cltv_avg"]
    )
    return ggf


def calculate_cltv(
    cltv_df: pd.DataFrame,
    bgf: BetaGeoFitter,
    ggf: GammaGammaFitter,
    months: int = 6,
    discount_rate: float = 0.01,
) -> pd.DataFrame:
    """Adim 3: BG/NBD + Gamma-Gamma modellerini birlestirerek `months` aylik
    CLTV'yi hesaplar ve `cltv` kolonu olarak ekler.
    """
    cltv = ggf.customer_lifetime_value(
        bgf,
        cltv_df["frequency"],
        cltv_df["recency_cltv_weekly"],
        cltv_df["T_weekly"],
        cltv_df["monetary_cltv_avg"],
        time=months,
        freq="W",
        discount_rate=discount_rate,
    )
    cltv_df["cltv"] = cltv.values
    return cltv_df


###############################################################
# GOREV 4: CLTV'ye Gore Segmentlerin Olusturulmasi
###############################################################

def assign_cltv_segments(cltv_df: pd.DataFrame) -> pd.DataFrame:
    """Adim 1: 6 aylik CLTV'ye gore musterileri D (en dusuk) - A (en yuksek)
    olacak sekilde 4 segmente ayirir.
    """
    cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
    return cltv_df


def cltv_segment_summary(cltv_df: pd.DataFrame) -> pd.DataFrame:
    """Adim 2: Segmentlerin recency, frequency, monetary ve cltv ortalamalari."""
    return cltv_df.groupby("cltv_segment", observed=True).agg(
        recency_haftalik=("recency_cltv_weekly", "mean"),
        frequency=("frequency", "mean"),
        monetary_ort=("monetary_cltv_avg", "mean"),
        cltv_ort=("cltv", "mean"),
        musteri_sayisi=("customer_id", "count"),
    ).sort_values("cltv_ort", ascending=False)


###############################################################
# BONUS: Tum Sureci Fonksiyonlastirma
###############################################################

def run_flo_cltv_pipeline(
    csv_path: str = DATA_PATH,
    output_dir: str = OUTPUT_DIR,
    months: int = 6,
):
    """Uctan uca CLTV pipeline'i: veri hazirlama -> cltv veri yapisi ->
    BG/NBD -> Gamma-Gamma -> cltv hesabi -> segmentasyon -> csv export.

    Returns
    -------
    dataframe : aykiri degerleri baskilanmis, toplam kolonlari eklenmis ham veri
    cltv_df   : customer_id bazli recency/frequency/monetary/cltv/segment tablosu
    """
    dataframe = data_prep(csv_path)
    cltv_df = create_cltv_df(dataframe)

    bgf = fit_bgnbd_model(cltv_df)
    ggf = fit_gamma_gamma_model(cltv_df)
    cltv_df = calculate_cltv(cltv_df, bgf, ggf, months=months)
    cltv_df = assign_cltv_segments(cltv_df)

    os.makedirs(output_dir, exist_ok=True)
    cltv_df.sort_values("cltv", ascending=False).to_csv(
        os.path.join(output_dir, "flo_cltv_predictions.csv"), index=False
    )

    return dataframe, cltv_df


if __name__ == "__main__":
    df = data_prep(DATA_PATH)
    cltv_df = create_cltv_df(df)

    bgf_model = fit_bgnbd_model(cltv_df)

    print("##################### 3. Ayda En Cok Satin Alim Beklenen Ilk 10 Musteri #####################")
    print(cltv_df.sort_values("exp_sales_3_month", ascending=False).head(10)[
        ["customer_id", "exp_sales_3_month"]
    ])

    print("##################### 6. Ayda En Cok Satin Alim Beklenen Ilk 10 Musteri #####################")
    print(cltv_df.sort_values("exp_sales_6_month", ascending=False).head(10)[
        ["customer_id", "exp_sales_6_month"]
    ])

    ggf_model = fit_gamma_gamma_model(cltv_df)
    cltv_df = calculate_cltv(cltv_df, bgf_model, ggf_model, months=6)

    print("##################### CLTV Degeri En Yuksek Ilk 20 Musteri #####################")
    print(cltv_df.sort_values("cltv", ascending=False).head(20)[
        ["customer_id", "frequency", "monetary_cltv_avg", "exp_average_value", "cltv"]
    ])

    cltv_df = assign_cltv_segments(cltv_df)

    print("##################### CLTV Segment Ozeti #####################")
    print(cltv_segment_summary(cltv_df))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cltv_df.sort_values("cltv", ascending=False).to_csv(
        os.path.join(OUTPUT_DIR, "flo_cltv_predictions.csv"), index=False
    )
    print(f"\nSonuclar kaydedildi: {os.path.join(OUTPUT_DIR, 'flo_cltv_predictions.csv')}")
