import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("sales_data_2020_2024.csv")
df["ORDERDATE"] = pd.to_datetime(df["ORDERDATE"])
df = df.sort_values("ORDERDATE")


# Analysis functions
def moving_average_analysis(df, window=3):
    df["MA"] = df["SALES"].rolling(window=window).mean()
    df["MA_diff"] = abs(df["SALES"] - df["MA"]) / df["MA"]
    df["MA_flag"] = df["MA_diff"] > 0.5
    return df


def z_score_analysis(df):
    df["Z_score"] = stats.zscore(df["SALES"])
    df["Z_flag"] = abs(df["Z_score"]) > 2
    return df


def iqr_analysis(df):
    Q1 = df["SALES"].quantile(0.25)
    Q3 = df["SALES"].quantile(0.75)
    IQR = Q3 - Q1
    df["IQR_flag"] = (df["SALES"] < (Q1 - 1.5 * IQR)) | (df["SALES"] > (Q3 + 1.5 * IQR))
    return df


def time_series_decomposition(df):
    daily_sales = df.set_index("ORDERDATE")["SALES"].resample("D").sum().fillna(0)
    if len(daily_sales) > 7:  # Need at least 2 periods for decomposition
        result = seasonal_decompose(daily_sales, model="additive", period=7)
        df["Trend"] = result.trend.reindex(df.index)
        df["Seasonal"] = result.seasonal.reindex(df.index)
        df["Residual"] = result.resid.reindex(df.index)
        df["Residual_flag"] = abs(df["Residual"]) > df["Residual"].std() * 2
    else:
        df["Trend"] = df["Seasonal"] = df["Residual"] = df["Residual_flag"] = np.nan
    return df


def cluster_analysis(df):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[["SALES"]])
    kmeans = KMeans(n_clusters=2, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X)
    smaller_cluster = df["Cluster"].value_counts().idxmin()
    df["Cluster_flag"] = df["Cluster"] == smaller_cluster
    return df


def benford_analysis(df):
    first_digits = df["SALES"].astype(str).str[0].astype(int)
    observed_freq = first_digits.value_counts().sort_index() / len(first_digits)
    expected_freq = pd.Series({d: np.log10(1 + 1 / d) for d in range(1, 10)})
    chi_square_stat = ((observed_freq - expected_freq) ** 2 / expected_freq).sum()
    p_value = 1 - stats.chi2.cdf(chi_square_stat, df=8)
    df["Benford_flag"] = p_value < 0.05
    return df, p_value


def isolation_forest_analysis(df):
    X = df[["SALES"]].values
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    df["Isolation_flag"] = iso_forest.fit_predict(X) == -1
    return df


def ewma_analysis(df, span=3):
    df["EWMA"] = df["SALES"].ewm(span=span).mean()
    df["EWMA_diff"] = abs(df["SALES"] - df["EWMA"]) / df["EWMA"]
    df["EWMA_flag"] = df["EWMA_diff"] > 0.5
    return df


def year_over_year_comparison(df):
    df["Year"] = df["ORDERDATE"].dt.year
    df["Month"] = df["ORDERDATE"].dt.month
    monthly_sales = df.groupby(["Year", "Month"])["SALES"].sum().unstack(level=0)
    yoy_change = (
        monthly_sales[monthly_sales.columns[-1]]
        / monthly_sales[monthly_sales.columns[-2]]
        - 1
    ) * 100
    return yoy_change


def combine_flags(df):
    flag_columns = [col for col in df.columns if col.endswith("_flag")]
    df["Total_flags"] = df[flag_columns].sum(axis=1)
    return df


# Apply all methods
df = moving_average_analysis(df)
df = z_score_analysis(df)
df = iqr_analysis(df)
df = time_series_decomposition(df)
df = cluster_analysis(df)
df, benford_p_value = benford_analysis(df)
df = isolation_forest_analysis(df)
df = ewma_analysis(df)
df = combine_flags(df)
yoy_change = year_over_year_comparison(df)

# Add year and month columns for analysis
df["Year"] = df["ORDERDATE"].dt.year
df["Month"] = df["ORDERDATE"].dt.month


# تحديث وظائف العرض المرئي
def plot_flag_distribution(df):
    plt.figure(figsize=(12, 6))
    sns.countplot(x="Total_flags", data=df)
    plt.title("توزيع إجمالي الإشارات لكل معاملة")
    plt.xlabel("عدد الإشارات")
    plt.ylabel("عدد المعاملات")
    plt.tight_layout()
    plt.show()


def plot_yearly_flag_comparison(df):
    yearly_flags = df.groupby("Year")["Total_flags"].sum().reset_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Year", y="Total_flags", data=yearly_flags)
    plt.title("إجمالي الإشارات حسب السنة")
    plt.xlabel("السنة")
    plt.ylabel("العدد الإجمالي للإشارات")
    plt.tight_layout()
    plt.show()


def plot_monthly_flag_heatmap(df):
    monthly_flags = df.groupby(["Year", "Month"])["Total_flags"].sum().unstack()
    plt.figure(figsize=(12, 8))
    sns.heatmap(monthly_flags, annot=True, fmt="d", cmap="YlOrRd")
    plt.title("خريطة حرارية للإشارات الشهرية")
    plt.xlabel("الشهر")
    plt.ylabel("السنة")
    plt.tight_layout()
    plt.show()


def plot_flag_method_comparison(df):
    flag_cols = [col for col in df.columns if col.endswith("_flag")]
    flag_counts = df[flag_cols].sum().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=flag_counts.index, y=flag_counts.values)
    plt.title("مقارنة طرق الإشارة")
    plt.xlabel("طريقة الإشارة")
    plt.ylabel("عدد الإشارات المرفوعة")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_sales_vs_flags_scatter(df):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x="SALES", y="Total_flags", data=df)
    plt.title("مبلغ المبيعات مقابل عدد الإشارات")
    plt.xlabel("مبلغ المبيعات")
    plt.ylabel("عدد الإشارات")
    plt.tight_layout()
    plt.show()


# إنشاء العروض المرئية المحسنة
plot_flag_distribution(df)
plot_yearly_flag_comparison(df)
plot_monthly_flag_heatmap(df)
plot_flag_method_comparison(df)
plot_sales_vs_flags_scatter(df)

# طباعة الإحصائيات الملخصة والمعاملات المشار إليها
print("الإحصائيات الملخصة:")
print(df["SALES"].describe())

print("\nقيمة p لقانون بنفورد:", benford_p_value)

print("\nالمعاملات المشار إليها:")
flagged_transactions = df[df["Total_flags"] > 0].sort_values(
    "Total_flags", ascending=False
)
print(flagged_transactions[["ORDERDATE", "SALES", "Total_flags"]])

print("\nالتغير الشهري في المبيعات من سنة لأخرى (%):")
print(yoy_change)

print("\nملخص الإشارات:")
print(f"إجمالي عدد المعاملات: {len(df)}")
print(f"عدد المعاملات المشار إليها: {df['Total_flags'].sum()}")
print(
    f"النسبة المئوية للمعاملات المشار إليها: {(df['Total_flags'].sum() / len(df)) * 100:.2f}%"
)

print("\nأعلى 5 معاملات مشار إليها:")
print(df.nlargest(5, "Total_flags")[["ORDERDATE", "SALES", "Total_flags"]])

print("\nعدد الإشارات الشهرية:")
print(df.groupby(["Year", "Month"])["Total_flags"].sum().unstack())
