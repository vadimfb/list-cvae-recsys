from typing import List, Dict, Union
import pandas as pd
import numpy as np

from src.utils.types import Slate


class PandasSlateGenerator:

    def __init__(self,
                 slate_size: int,
                 num_items: int,
                 max_number_of_slates_per_user: int = 3,
                 min_positive_item_in_slate: int = 1,
                 seed: int = 0):

        self.slate_size = slate_size
        self.num_items = num_items
        self.max_number_of_slates_per_user = max_number_of_slates_per_user
        self.min_positive_item_in_slate = min_positive_item_in_slate

        np.random.seed(seed)

    def _generate_slate(self, user_id: Union[int, str], items: List[int], num_negatives: int) -> Slate:

        num_positives = self.slate_size - num_negatives
        if num_positives >= len(items):
            num_positives = len(items)
            if num_positives > 1:
                num_positives -= 1
            num_negatives = self.slate_size - num_positives

        history = np.array(items[:-num_positives])
        positives = items[-num_positives:]
        negatives = np.random.randint(0, self.num_items - 1, num_negatives).tolist()
        slate_items = positives + negatives
        responses = [1] * num_positives + [0] * num_negatives

        items_with_response = list(zip(slate_items, responses))
        np.random.shuffle(items_with_response)
        slate_items = np.array([item for item, _ in items_with_response])
        responses = np.array([response for _, response in items_with_response])

        return Slate(user_id=user_id, slate_items=slate_items, slate_responses=responses, user_history_items=history)

    def generate_train_slates(self,
                              train_df: pd.DataFrame,
                              user_col: str,
                              item_col: str,
                              timestamp_col: str) -> List[Slate]:

        user_items = (
            train_df
            .sort_values([user_col, timestamp_col])
            .groupby(user_col)[item_col]
            .agg(list)
            .reset_index()
            .rename(columns={item_col: 'items'})
        )

        n_users = user_items.shape[0]

        user_items['num_items'] = user_items['items'].apply(len)
        nums_of_negatives = np.random.randint(low=0,
                                              high=self.slate_size - self.min_positive_item_in_slate,
                                              size=(n_users, self.max_number_of_slates_per_user))
        user_items['num_negatives'] = nums_of_negatives.tolist()

        slates = []
        for row in user_items.itertuples():
            user_id = getattr(row, user_col)
            n_items = row.num_items
            items = row.items
            num_negatives = row.num_negatives

            slate = self._generate_slate(user_id, items, num_negatives[0])
            slates.append(slate)

            i = 1
            while n_items > (i + 1) * self.slate_size:
                if (i + 1) > self.max_number_of_slates_per_user:
                    break
                slate = self._generate_slate(user_id, items[: i * self.slate_size], num_negatives[i])
                slates.append(slate)
                i += 1

        return slates

    @staticmethod
    def generate_test_slates(train_df: pd.DataFrame,
                             test_df: pd.DataFrame,
                             user_col: str,
                             item_col: str) -> List[Slate]:

        user_train_items = (
            train_df[train_df[user_col].isin(test_df[user_col].unique())]
            .groupby(user_col)[item_col]
            .agg(list)
            .to_dict()
        )

        user_test_items = (
            test_df
            .groupby(user_col)[item_col]
            .agg(list)
            .to_dict()
        )

        slates = []
        for user_id, true_items in user_test_items.items():
            history_items = user_train_items.get(user_id, [])
            slate = Slate(user_id=user_id,
                          slate_items=np.array(true_items),
                          slate_responses=np.ones_like(true_items),
                          user_history_items=np.array(history_items))
            slates.append(slate)

        return slates

    def save_slates_to_csv(self, slates: List[Slate], path_to_slates: str):

        df = pd.DataFrame.from_records((self._slate_to_save_format(slate) for slate in slates))
        df.to_csv(path_to_slates, index=False)

    @staticmethod
    def _slate_to_save_format(slate: Slate) -> Dict[str, str]:
        slate_vars = vars(slate)

        for field_name, values in slate_vars.items():
            if field_name != 'user_id':
                slate_vars[field_name] = '|'.join(map(str, values))

        return slate_vars
