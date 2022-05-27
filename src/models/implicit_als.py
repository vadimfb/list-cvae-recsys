from typing import Dict
import pandas as pd
import numpy as np
from implicit.als import AlternatingLeastSquares
import scipy.sparse as sp
from torch.utils.data import DataLoader


class ImplicitALS:

    def __init__(self,
                 num_users: int,
                 num_items: int,
                 embedding_dim: int,
                 alpha: float = 40.,
                 regularization: float = 0.01,
                 epochs: int = 15):

        self.num_users = num_users
        self.num_items = num_items

        self.alpha = alpha

        self.model = AlternatingLeastSquares(factors=embedding_dim, regularization=regularization, iterations=epochs)
        self.train_matrix = None

    def _prepare_train_matrix(self, row_interactions: pd.DataFrame) -> sp.csr.csr_matrix:
        inverse_shape = (self.num_items, self.num_users)

        train_matrix = sp.csr_matrix(
            (
                np.ones(row_interactions.shape[0]), (row_interactions['item'], row_interactions['user'])
            ),
            shape=inverse_shape
        )

        self.train_matrix = train_matrix
        return train_matrix

    def train_implicit_als(self, row_interactions: pd.DataFrame):

        train_matrix = self._prepare_train_matrix(row_interactions)
        train_matrix.data = self.alpha * train_matrix.data + 1

        self.model.fit(train_matrix, show_progress=True)

    def recommend(self, test_loader: DataLoader, topk: int = 10) -> Dict:
        recommendations = {}
        user_items = self.train_matrix.T.tocsr()
        for user_id, _, _, _, _, _ in test_loader:
            user_id = user_id.numpy().ravel()
            ids = []
            for user in user_id:
                recs = self.model.recommend(user,
                                            user_items,
                                            N=topk,
                                            filter_already_liked_items=True,
                                            filter_items=None,
                                            recalculate_user=False)
                ids.append([item for item, _ in recs])

            recommendations.update(zip(user_id, map(list, ids)))

        return recommendations
