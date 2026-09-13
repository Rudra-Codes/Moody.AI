import pandas as pd

def stratified_sample(data, n, stratefied_col="label", random_state=42):
    df = data.copy()
    if n >= len(df):
        return df

    frac = n / len(df)

    sampled = (
        df.groupby(stratefied_col, group_keys=False).sample(frac=frac, random_state=random_state)
    )

    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=random_state)
    elif len(sampled) < n:
        remaining = df.drop(sampled.index)
        extra = remaining.sample(n=n - len(sampled), random_state=random_state)
        sampled = pd.concat([sampled, extra])

    return sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)