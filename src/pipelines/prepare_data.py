from typing import Dict
from argparse import ArgumentParser
import src.data_extraction.data_readers as readers
import src.data_preparation.mappers as mappers
from src.data_preparation.splitters import PandasTimestampSplitter
from src.data_preparation.slate_generators import PandasSlateGenerator
from src.utils.json_utils import write_json
from src.utils.config_utils import read_config


def main(config: Dict):

    reader_name = config['reader_name']
    reader_params = config.get('reader_params', {})
    reader = getattr(readers, reader_name)(**reader_params)
    df, df_info = reader.read()

    item_mapper_name = config['item_mapper']
    item_mapper_params = config.get('item_mapper_params', {})
    mapper = getattr(mappers, item_mapper_name)(column=reader.item_col, **item_mapper_params)
    mapper.map(df, 'to_new')
    mapper.save(config['path_to_item_mapper'])

    user_mapper_name = config['user_mapper']
    user_mapper_params = config.get('user_mapper_params', {})
    mapper = getattr(mappers, user_mapper_name)(column=reader.user_col, **user_mapper_params)
    mapper.map(df, 'to_new')
    mapper.save(config['path_to_user_mapper'])

    # Save df_info
    write_json(df_info, config['path_to_data_info'])

    splitter = PandasTimestampSplitter(test_size=config['test_size'])
    train_df, test_df = splitter.split(df, user_col=reader.user_col, timestamp_col=reader.timestamp_col)

    # Save train/test row interactions
    train_df.to_csv(config['train_row_interactions_path'], index=False)
    test_df.to_csv(config['test_row_interactions_path'], index=False)

    slate_generator = PandasSlateGenerator(slate_size=config['slate_size'],
                                           num_items=df_info['num_items'],
                                           max_number_of_slates_per_user=config['max_number_of_slates_per_user'])

    train_slates = slate_generator.generate_train_slates(train_df,
                                                         user_col=reader.user_col,
                                                         item_col=reader.item_col,
                                                         timestamp_col=reader.timestamp_col)
    test_slates = slate_generator.generate_test_slates(train_df=train_df,
                                                       test_df=test_df,
                                                       user_col=reader.user_col,
                                                       item_col=reader.item_col)

    slate_generator.save_slates_to_csv(train_slates, path_to_slates=config['path_to_train_slates'])
    slate_generator.save_slates_to_csv(test_slates, path_to_slates=config['path_to_test_slates'])


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--config', default='configs/config.yml')
    args = arg_parser.parse_args()
    config = read_config(args.config, mode='prepare_data')
    main(config)
