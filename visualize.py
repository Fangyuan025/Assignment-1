import pandas as pd
import matplotlib.pyplot as plt


def plot_counts_and_proportions(df, dataset_name, axes_pair):
    """
    df: pandas DataFrame with a column 'negativereason'
    dataset_name: string label for the dataset (e.g., 'All Tweets')
    axes_pair: tuple (ax_counts, ax_proportions) from Matplotlib subplots
    """
    ax_counts, ax_proportions = axes_pair

    counts = df["negativereason"].value_counts()
    proportions = df["negativereason"].value_counts(normalize=True)

    counts.plot(kind='bar', color='skyblue', ax=ax_counts, rot=0)
    ax_counts.set_title(f"{dataset_name} - Negative Reason Counts")
    ax_counts.set_xlabel("Negative Reason")
    ax_counts.set_ylabel("Count")

    for i, v in enumerate(counts):
        ax_counts.text(i, v + 0.5, str(v), ha='center')

    proportions.plot(kind='bar', color='orange', ax=ax_proportions, rot=0)
    ax_proportions.set_title(f"{dataset_name} - Negative Reason Proportions")
    ax_proportions.set_xlabel("Negative Reason")
    ax_proportions.set_ylabel("Proportion")

    for i, v in enumerate(proportions):
        ax_proportions.text(i, v + 0.01, f"{v:.2f}", ha='center')


def main():

    df_combined = pd.read_csv("tweets_preprocessed.csv", encoding="utf-8")

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    plot_counts_and_proportions(df_combined, "All Tweets", (axes[0], axes[1]))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()