# -*- coding: utf-8 -*-

import pandas as pd
import re

# 1. Read the data

df = pd.read_csv("Tweets.csv", encoding="utf-8", sep=",")

# 2. Keep only the two columns we need: "text" (tweet content) and "negativereason" (negative reason)
df = df[["text", "negativereason"]]

# 3. Retain rows that have negative reasons in the following 5 categories:
#    "Bad Flight", "Late Flight", "Customer Service Issue", "Cancelled Flight", "Flight Booking Problems"
valid_reasons = [
    "Bad Flight",
    "Late Flight",
    "Customer Service Issue",
    "Cancelled Flight",
    "Flight Booking Problems"
]
df = df[df["negativereason"].isin(valid_reasons)]

# 4. Remove usernames that follow '@' from the "text" column
df["text"] = df["text"].apply(lambda x: re.sub(r"@\S+", "", str(x)))

# 5. Reset the index to keep it continuous
df.reset_index(drop=True, inplace=True)

df.to_csv("tweets_preprocessed.csv", index=False, encoding="utf-8")

print(df.head(10))