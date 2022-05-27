from typing import List, Tuple, Dict
import pandas as pd


class BaseReader:

    def __init__(self,
                 source_path: str,
                 user_col: str,
                 item_col: str,
                 rating_col: str,
                 timestamp_col: str,
                 **kwargs):
        self.source_path = source_path
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.timestamp_col = timestamp_col

    def read(self) -> Tuple[pd.DataFrame, Dict]:
        df = self.load()
        df_info = self._get_df_info(df)
        self._rename_cols(df)
        return df, df_info

    def load(self) -> pd.DataFrame:
        pass

    def _get_df_info(self, df: pd.DataFrame) -> Dict:
        num_users = df[self.user_col].nunique()
        num_items = df[self.item_col].nunique()
        num_interactions = df.shape[0]

        return dict(num_users=num_users, num_items=num_items, num_interactions=num_interactions)

    def _rename_cols(self, df: pd.DataFrame):
        columns = {
            self.user_col: 'user',
            self.item_col: 'item',
            self.rating_col: 'rating',
            self.timestamp_col: 'ts'
        }
        df.rename(columns=columns, inplace=True)
        self.user_col = 'user'
        self.item_col = 'item'
        self.rating_col = 'rating'
        self.timestamp_col = 'ts'


class BasePandasReader(BaseReader):

    def __init__(self,
                 source_path: str,
                 user_col: str,
                 item_col: str,
                 rating_col: str,
                 timestamp_col: str,
                 additional_cols: List[str] = None):

        super(BasePandasReader, self).__init__(source_path,
                                               user_col,
                                               item_col,
                                               rating_col,
                                               timestamp_col)
        self.additional_cols = []
        if additional_cols:
            self.additional_cols = additional_cols

        main_cols = [self.user_col, self.item_col, self.rating_col, self.timestamp_col]
        self.use_cols = main_cols + self.additional_cols

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.source_path,
                         usecols=self.use_cols,
                         dtype={self.user_col: 'uint32', self.item_col: 'uint32', self.rating_col: 'float32'})
        return df


class MovielensDataReader(BasePandasReader):

    def __init__(self,
                 source_path: str,
                 user_col: str = 'userid',
                 item_col: str = 'movieid',
                 rating_col: str = 'rating',
                 timestamp_col: str = 'timestamp',
                 additional_cols: List[str] = None):
        
        super(MovielensDataReader, self).__init__(source_path,
                                                  user_col,
                                                  item_col,
                                                  rating_col,
                                                  timestamp_col,
                                                  additional_cols)

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.source_path,
                         engine='python',
                         sep='::',
                         header=None,
                         names=[self.user_col, self.item_col, self.rating_col, self.timestamp_col])
        return df
