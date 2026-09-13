# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "bs4>=0.0.2",
#     "pandas>=3.0.5",
#     "pyarrow",
# ]
# ///

import pandas as pd
from bs4 import BeautifulSoup

DATA_DIR = "/home/khush/coding/ds_proj/test-00000-of-00001.parquet"


def clean_html(value):
    if pd.isna(value):
        return value
    return BeautifulSoup(str(value), "html.parser").get_text(" ", strip=True)


def update_table(df):
    df.to_csv("test.csv", index=False)


if __name__ == "__main__":
    df = pd.read_parquet(DATA_DIR)

    # Clean first column
    df.iloc[:, 0] = df.iloc[:, 0].apply(clean_html)

    update_table(df)