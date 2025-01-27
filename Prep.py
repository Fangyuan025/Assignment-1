# -*- coding: utf-8 -*-

import pandas as pd
import re

# 1. Read the data
# Assume the original data filename is "Tweets.txt" and located in the same directory as this script.
# If the file is not in the current directory, please adjust the file path accordingly.
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
#    We use a regular expression to remove all occurrences of '@' followed by non-whitespace characters
df["text"] = df["text"].apply(lambda x: re.sub(r"@\S+", "", str(x)))

# 5. Reset the index to keep it continuous
df.reset_index(drop=True, inplace=True)

# 6. (Optional) If you need to save the cleaned data to a file, use the code below. Otherwise, you can comment it out.
df.to_csv("tweets_preprocessed.csv", index=False, encoding="utf-8")

# The resulting DataFrame df now contains only rows with the selected five negative reasons,
# includes only the "text" and "negativereason" columns, and has removed any occurrences of @username in the text.
print(df.head(10))