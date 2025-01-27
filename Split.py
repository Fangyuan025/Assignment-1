#!/usr/bin/env python3
# split.py
# This script reads the preprocessed data (tweets_preprocessed.csv)
# then splits it into training (80%) and testing (20%) sets.
# We use stratified sampling to ensure that the proportion of
# each negative reason category remains consistent across both sets.

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    # 1. Read the preprocessed dataset
    df = pd.read_csv("tweets_preprocessed.csv", encoding="utf-8")

    # 2. Perform an 80/20 split with stratification on "negativereason"
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,  # for reproducibility
        stratify=df["negativereason"]
    )

    # 3. Save the resulting training and testing sets to separate CSV files
    train_df.to_csv("training.csv", index=False, encoding="utf-8")
    test_df.to_csv("testing.csv", index=False, encoding="utf-8")

    print("Data has been split into training.csv (80%) and testing.csv (20%).")
    print("Stratification ensured that the proportion of each category remains consistent.")


if __name__ == "__main__":
    main()
