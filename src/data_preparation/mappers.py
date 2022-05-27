import numpy as np
import pandas as pd
import pickle


class BaseMapper:

    def __init__(self, column: str, mapping: pd.DataFrame = None, **kwargs):
        self.column = column
        self.mapping = mapping

    def save(self, path_to_mapper: str):

        with open(path_to_mapper, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path_to_mapper: str):

        with open(path_to_mapper, 'rb') as f:
            mapper = pickle.load(f)

        return mapper


class PandasColumnMapper(BaseMapper):

    def __init__(self, column: str, mapping: pd.DataFrame = None):
        super(PandasColumnMapper, self).__init__(column=column, mapping=mapping)
        self.map_to_new = None
        self.map_to_old = None

    def _set_maps(self):
        self.map_to_new = self.mapping.set_index('old')['new'].to_dict()
        self.map_to_old = self.mapping.set_index('new')['old'].to_dict()

    def _get_mapping(self, df: pd.DataFrame):
        uniq_values = df[self.column].unique()
        n_uniqs = len(uniq_values)
        mapping = pd.DataFrame({
            'old': uniq_values,
            'new': np.arange(n_uniqs, dtype=np.uint32)
        })
        self.mapping = mapping
        self._set_maps()

    def map(self, df: pd.DataFrame, map_type: str):

        if self.mapping is None:
            self._get_mapping(df)

        if map_type == 'to_new':
            df[self.column] = df[self.column].map(self.map_to_new)
        else:
            df[self.column] = df[self.column].map(self.map_to_old)
