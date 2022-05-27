from typing import Dict, Any, List
from collections import defaultdict
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import src.evaluation.metrics as metrics


class Evaluator:

    def __init__(self, models: Dict[str, Any], metric_names: List[str], topk: int):

        self.models = models
        self.topk = topk
        self.metric_names = metric_names

    def evaluate(self, test_dataset: Dataset) -> pd.DataFrame:

        test_loader = DataLoader(test_dataset)
        result = defaultdict(list)
        true_item_lists = self._get_true_item_lists(test_loader)
        for model_name, model in self.models.items():
            recommendations = model.recommend(test_loader, topk=self.topk)
            for metric_name in self.metric_names:
                metric_func = getattr(metrics, metric_name)
                metric_val = metric_func(true_item_lists, recommendations, self.topk)
                result[model_name].append(metric_val)

        result = pd.DataFrame(result)
        result.index = self.metric_names

        return result

    @staticmethod
    def _get_true_item_lists(test_loader: DataLoader) -> Dict:
        relevant_items = {}
        with torch.no_grad():
            for user_id, slate_items, _, _, slate_lens, _ in test_loader:

                slate_items = slate_items.numpy()
                slate_lens = slate_lens.numpy()
                true_slates = []
                for slate, slate_len in zip(slate_items, slate_lens):
                    true_slates.append(slate[:slate_len].tolist())

                users = user_id.numpy()
                relevant_items.update(zip(users, true_slates))

        return relevant_items
