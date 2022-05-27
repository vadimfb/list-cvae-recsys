from typing import Dict
from argparse import ArgumentParser
from src.utils.json_utils import read_json
from src.pipelines.trainers import ListCVAETrainer
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

    train_results, test_results, random_results = trainer.train_eval_model(train_dataset, test_dataset)
    trainer.plot_train_results(train_results, config['train_curve_fig'])
    trainer.plot_test_results(test_results, config['eval_curve_fig'])


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--config', default='configs/config.yml')
    args = arg_parser.parse_args()
    config = read_config(args.config, mode='train_eval')
    main(config)
