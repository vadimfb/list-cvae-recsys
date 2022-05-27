from typing import Tuple
import pandas as pd


class PandasTimestampSplitter:

    def __init__(self, test_size: float):

        self.test_size = test_size

    def split(self, df: pd.DataFrame, user_col: str, timestamp_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:

        df = df.sort_values(timestamp_col).reset_index(drop=True)

        train_last_idx = int(df.shape[0] * (1 - self.test_size))

        train_df = df.loc[:train_last_idx]
        train_users = train_df[user_col].unique()
        test_df = df.loc[train_last_idx:]
        test_df = test_df[test_df[user_col].isin(train_users)]

        return train_df, test_df
