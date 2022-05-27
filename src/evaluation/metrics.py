from typing import List, Dict, Union
from math import log2


def precision_at_k(relevant_lists: Dict[Union[int, str], List[int]],
                   recommend_lists: Dict[Union[int, str], List[int]],
                   k: int = 10) -> float:

    hits = 0
    n_recommendations = 0

    for user, relevant_items in relevant_lists.items():

        recommend_items = recommend_lists.get(user, [])
        recommend_items = recommend_items[: k]
        intersection_items = set(relevant_items) & set(recommend_items)
        hits += len(intersection_items)
        n_recommendations += k

    precision = hits / n_recommendations

    return precision


def recall_at_k(relevant_lists: Dict[Union[int, str], List[int]],
                recommend_lists: Dict[Union[int, str], List[int]],
                k: int = 10) -> float:
    hits = 0
    n_relevants = 0

    for user, relevant_items in relevant_lists.items():
        recommend_items = recommend_lists.get(user, [])
        recommend_items = recommend_items[: k]
        intersection_items = set(relevant_items) & set(recommend_items)
        hits += len(intersection_items)
        n_relevants += min(len(relevant_items), k)

    recall = hits / n_relevants

    return recall


def average_precision_at_k(relevant_items: List[int], recommend_items: List[int], k: int = 10) -> float:

    relevant_items = set(relevant_items)

    score = 0
    for rank, item in enumerate(recommend_items[: k]):
        if item in relevant_items:
            intersection_items = relevant_items.intersection(recommend_items[: rank + 1])
            hits = len(intersection_items)
            p_at_rank = hits / (rank + 1)
            score += p_at_rank

    return score / min(k, len(relevant_items))


def mean_average_precision_at_k(relevant_lists: Dict[Union[int, str], List[int]],
                                recommend_lists: Dict[Union[int, str], List[int]],
                                k: int = 10) -> float:

    ap_sum = 0
    n_users = 0
    for user, relevant_items in relevant_lists.items():
        recommend_items = recommend_lists.get(user, [])
        recommend_items = recommend_items[: k]
        ap_sum += average_precision_at_k(relevant_items, recommend_items, k)
        n_users += 1

    mean_average_precision = ap_sum / n_users

    return mean_average_precision


def normalized_discounted_cumulative_gain_at_k(relevant_lists: Dict[Union[int, str], List[int]],
                                               recommend_lists: Dict[Union[int, str], List[int]],
                                               k: int = 10) -> float:

    ndcg = 0
    n_users = 0

    for user, relevant_items in relevant_lists.items():
        recommend_items = recommend_lists.get(user, [])
        recommend_items = recommend_items[: k]
        dcg = 0
        idcg = 0
        for rank, item in enumerate(recommend_items):
            if item in relevant_items:
                dcg += 1 / log2(rank + 1 + 1)
            idcg += 1 / log2(rank + 1 + 1)

        ndcg += dcg / idcg
        n_users += 1

    ndcg = ndcg / n_users

    return ndcg
