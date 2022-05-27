import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class SlateFormationDataset(Dataset):
    def __init__(self, path_to_slates: str, num_items: int):
        self.slate_formations = pd.read_csv(path_to_slates)
        self.num_items = num_items

        self.users = None
        self.slates = None
        self.response_vectors = None
        self.user_history_items = None
        self.max_user_history_len = None
        self.max_user_slate_len = None

        self.convert_to_vector_form()

    def convert_to_vector_form(self):

        def to_int64_numpy_array(array):
            if array is not None and array is not np.nan:
                return np.array(array, dtype=np.int64)
            return np.array([], dtype=np.int64)

        self.slates = (
            self.slate_formations['slate_items']
            .str.split('|')
            .apply(to_int64_numpy_array)
        )
        self.response_vectors = (
            self.slate_formations['slate_responses']
            .str.split('|')
            .apply(to_int64_numpy_array)
        )
        self.user_history_items = (
            self.slate_formations['user_history_items']
            .str.split('|')
            .apply(to_int64_numpy_array)
        )

        self.max_user_slate_len = self.slates.apply(len).max()
        self.max_user_history_len = self.user_history_items.apply(len).max()

        self.users = self.slate_formations['user_id'].to_dict()
        self.slates = self.slates.to_dict()
        self.response_vectors = self.response_vectors.to_dict()
        self.user_history_items = self.user_history_items.to_dict()

    def __len__(self):
        return self.slate_formations.shape[0]

    def __getitem__(self, idx):

        user_id = self.users[idx]
        slate_items = np.full(self.max_user_slate_len, self.num_items)
        slate_items[:len(self.slates[idx])] = self.slates[idx]
        slate_responses = np.zeros(self.max_user_slate_len)
        slate_responses[: len(self.slates[idx])] = self.response_vectors[idx]

        user_history_items = np.full(self.max_user_history_len, self.num_items)
        user_history_items[:len(self.user_history_items[idx])] = self.user_history_items[idx]

        len_slate = len(self.slates[idx])
        len_user_history = max(len(self.user_history_items[idx]), 1)

        return user_id, slate_items, slate_responses, user_history_items, len_slate, len_user_history
