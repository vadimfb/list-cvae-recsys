from typing import Dict
from argparse import ArgumentParser
import pandas as pd
from src.utils.json_utils import read_json
from src.pipelines.trainers import ListCVAETrainer
from src.models.implicit_als import ImplicitALS
from src.evaluation.evaluators import Evaluator
from src.data_preparation.datasets import SlateFormationDataset
from src.utils.config_utils import read_config


def main(config: Dict):

    dataset_info = read_json(config['dataset_info_path'])
    model_parameters = config['model_parameters']
    model_parameters['num_items'] = dataset_info['num_items']
    train_parameters = config['train_parameters']
    optimizer_parameters = config['optimizer_parameters']

    trainer = ListCVAETrainer(optimizer_parameters=optimizer_parameters,
                              model_parameters=model_parameters,
                              train_parameters=train_parameters)

    train_dataset = SlateFormationDataset(config['path_to_train_slates'], dataset_info['num_items'])
    test_dataset = SlateFormationDataset(config['path_to_test_slates'], dataset_info['num_items'])

    trainer.train_model(train_dataset)

    num_items = dataset_info['num_items']
    num_users = dataset_info['num_users']
    als_params = config['als_params']
    als = ImplicitALS(num_users=num_users, num_items=num_items, **als_params)
    row_interactions = pd.read_csv(config['train_row_interactions_path'])
    als.train_implicit_als(row_interactions)

    evaluator = Evaluator(models={'List-CVAE': trainer.model, 'ImplicitALS': als},
                          metric_names=config['metric_names'],
                          topk=config['topk'])
    results = evaluator.evaluate(test_dataset)
    results.to_csv(config['eval_results_path'], index=True)


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--config', default='configs/config.yml')
    args = arg_parser.parse_args()
    config = read_config(args.config, mode='evaluate')
    main(config)
