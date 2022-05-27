from typing import Dict
import yaml


def read_config(path: str, mode: str) -> Dict:
    """The method to read config

    Args:
        path (str): path
        mode (str): name of running pipeline

    Returns:
        Dict: config

    """
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config[mode]
