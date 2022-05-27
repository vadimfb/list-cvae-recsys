from typing import Union
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Slate:
    user_id: Union[int, str]
    slate_items: np.ndarray
    slate_responses: np.ndarray
    user_history_items: np.ndarray
