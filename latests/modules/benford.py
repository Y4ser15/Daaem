import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def get_first_digit(value):
    try:
        str_value = str(abs(value)).lstrip("0")
        if str_value and str_value[0].isdigit():
            return int(str_value[0])
    except:
        pass
    return None


def test_benfords_law(file_path, column_name, margin_of_error=0.05):
    data = pd.read_csv(file_path)
    first_digits = data[column_name].apply(get_first_digit).dropna()
    observed = first_digits.value_counts().sort_index()
    total_valid = len(first_digits)

    expected_proportions = np.log10(1 + 1 / np.arange(1, 10))
    expected = total_valid * expected_proportions

    # Ensure observed and expected have the same index
    full_range = pd.Series(index=range(1, 10), dtype=float)
    observed = observed.reindex(full_range.index, fill_value=0)

    observed_prop = observed / total_valid

    # Calculate chi-square and p-value
    chi_square, p_value = stats.chisquare(f_obs=observed, f_exp=expected)

    # Prepare data for visualization
    results_df = pd.DataFrame(
        {
            "Digit": range(1, 10),
            "Observed": observed_prop,
            "Expected": expected_proportions,
            "Difference": observed_prop - expected_proportions,
        }
    )

    # Create the visualization
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 14), gridspec_kw={"height_ratios": [3, 1]}
    )

    # First plot (original bar chart)
    x = np.arange(1, 10)
    width = 0.35

    ax1.bar(x - width / 2, observed_prop, width, alpha=0.5, label="Observed")
    ax1.bar(
        x + width / 2,
        expected_proportions,
        width,
        alpha=0.5,
        label="Expected (Benford's Law)",
    )

    ax1.set_xlabel("First Digit", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title(f"Benford's Law Test for {column_name}", fontsize=16)
    ax1.set_xticks(x)
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    for i, (obs, exp) in enumerate(zip(observed_prop, expected_proportions)):
        ax1.text(
            x[i] - width / 2, obs, f"{obs:.3f}", ha="center", va="bottom", fontsize=9
        )
        ax1.text(
            x[i] + width / 2, exp, f"{exp:.3f}", ha="center", va="bottom", fontsize=9
        )

    # Second plot (difference chart)
    sns.barplot(
        x="Digit",
        y="Difference",
        data=results_df,
        ax=ax2,
        palette=[
            "red" if abs(d) > margin_of_error else "green"
            for d in results_df["Difference"]
        ],
    )
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax2.axhline(y=margin_of_error, color="red", linestyle="--", linewidth=0.5)
    ax2.axhline(y=-margin_of_error, color="red", linestyle="--", linewidth=0.5)

    ax2.set_title("Difference (Observed - Expected)", fontsize=14)
    ax2.set_xlabel("First Digit", fontsize=12)
    ax2.set_ylabel("Difference", fontsize=12)

    # Add value labels on the difference bars
    for i, diff in enumerate(results_df["Difference"]):
        ax2.text(i, diff, f"{diff:.3f}", ha="center", va="center", fontsize=9)

    plt.tight_layout()
    plt.show()

    # Print analysis results
    print(f"Benford's Law Test Results for {column_name}:")
    print(
        f"\nDigits within {margin_of_error*100}% margin of error: {sum(abs(results_df['Difference']) <= margin_of_error)} out of 9"
    )
    print("Deviation from expected:")
    for _, row in results_df.iterrows():
        deviation = row["Difference"]
        print(
            f"Digit {row['Digit']}: {deviation:.4f} ({'within' if abs(deviation) <= margin_of_error else 'outside'} margin)"
        )

    if p_value > 0.05:
        print(
            "\nStatistically, the data follows Benford's Law (fail to reject null hypothesis)"
        )
    else:
        print(
            "\nStatistically, the data does not follow Benford's Law (reject null hypothesis)"
        )

    if sum(abs(results_df["Difference"]) <= margin_of_error) >= 7:
        print(
            f"Considering a {margin_of_error*100}% margin of error, the data generally follows Benford's Law"
        )
    else:
        print(
            f"Even with a {margin_of_error*100}% margin of error, the data shows significant deviations from Benford's Law"
        )


if __name__ == "__main__":
    file_path = "data/fake.csv"
    column_name = "SALES"
    test_benfords_law(file_path, column_name, margin_of_error=0.05)
