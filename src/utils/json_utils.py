from typing import Dict
import json


def read_json(path: str) -> Dict:
    with open(path, 'r') as f:
        dictionary = json.load(f)
    return dictionary


def write_json(dictionary: Dict, path: str):
    with open(path, 'w') as f:
        json.dump(dictionary, f)