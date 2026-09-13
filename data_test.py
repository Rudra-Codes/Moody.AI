import pandas as pd

DATASET_PATH = "data/dataset_train.csv"
df = pd.read_csv(DATASET_PATH)
print(df.head())

print(df['text'][0], df['label'][0])