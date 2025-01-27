#!/usr/bin/env python3
# visualize.py
# This script plots figures (counts & proportions) for:
# 1) the combined preprocessed dataset (tweets_preprocessed.csv) -- optional if you'd like to keep it
# 2) the training set (training.csv)
# 3) the testing set (testing.csv)
#
# Note: If you'd like to remove the original dataset's plotting, feel free to comment or delete that part.

import pandas as pd
import matplotlib.pyplot as plt


def plot_counts_and_proportions(df, dataset_name, axes_pair):
    """
    df: pandas DataFrame with a column 'negativereason'
    dataset_name: string label for the dataset (e.g., 'Training')
    axes_pair: tuple (ax_counts, ax_proportions) from Matplotlib subplots
    """
    ax_counts, ax_proportions = axes_pair

    counts = df["negativereason"].value_counts()
    proportions = df["negativereason"].value_counts(normalize=True)

    # --- Plot counts ---
    counts.plot(kind='bar', color='skyblue', ax=ax_counts, rot=0)
    ax_counts.set_title(f"{dataset_name} - Negative Reason Counts")
    ax_counts.set_xlabel("Negative Reason")
    ax_counts.set_ylabel("Count")
    for i, v in enumerate(counts):
        ax_counts.text(i, v + 0.5, str(v), ha='center')

    # --- Plot proportions ---
    proportions.plot(kind='bar', color='orange', ax=ax_proportions, rot=0)
    ax_proportions.set_title(f"{dataset_name} - Negative Reason Proportions")
    ax_proportions.set_xlabel("Negative Reason")
    ax_proportions.set_ylabel("Proportion")
    for i, v in enumerate(proportions):
        ax_proportions.text(i, v + 0.01, f"{v:.2f}", ha='center')


def main():
    # 1. Read the processed (combined) dataset (optional)
    #    If you'd rather skip plotting the combined data, you can comment this out.
    df_combined = pd.read_csv("tweets_preprocessed.csv", encoding="utf-8")

    # 2. Read the training and testing sets
    df_train = pd.read_csv("training.csv", encoding="utf-8")
    df_test = pd.read_csv("testing.csv", encoding="utf-8")

    # 3. Set up the figure with subplots
    #    We'll create 2 rows x 2 columns for training sets, then 2 rows x 2 columns for testing sets
    #    Alternatively, we can do a single 2x2 with training (top) and testing (bottom).
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    # 4. Plot: Training set (top row), Testing set (bottom row)
    #    axes[0,0] and axes[0,1] for training (counts & proportions)
    #    axes[1,0] and axes[1,1] for testing (counts & proportions)
    plot_counts_and_proportions(df_train, "Training", (axes[0, 0], axes[0, 1]))
    plot_counts_and_proportions(df_test, "Testing", (axes[1, 0], axes[1, 1]))

    plt.tight_layout()
    plt.show()

    # --- If you'd also like to plot the combined data in a separate figure, uncomment below: ---
    """
    fig_combined, axes_combined = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    plot_counts_and_proportions(df_combined, "Combined (All Data)", (axes_combined[0], axes_combined[1]))
    plt.tight_layout()
    plt.show()
    """


if __name__ == "__main__":
    main()
