# List-CVAE repa

## Repa structure
    DS-223-conditional-cvae-model
    ├── src
    │   ├── data_extraction
    │   │   ├── data_readers.py                     # module of data loading
    │   │   └── __init__.py
    │   ├── data_preparation
    │   │   ├── datasets.py                         # module with custom datasets
    │   │   ├── mappers.py                          # mappers user/item_id -> idxs
    │   │   ├── slate_generators.py                 # generators of "slate" for List-CVAE
    │   │   ├── splitters.py                        # train/test splitters
    │   │   └── __init__.py
    │   ├── models
    │   │   ├── list_cvae.py                        # List-CVAE model
    │   │   ├── implicit_als.py                     # Implicit ALS model by implicit framework
    │   │   └── __init__.py
    │   ├── model_selection                         # TO DO
    │   │   └── ...
    │   ├── evaluation
    │   │   ├── evaluators.py                       # module of handlers for compare models
    │   │   └── metrics.py                          # recsys metrics
    │   ├── pipelines
    │   │   ├── prepare_data.py                     # data preparation pipeline for experiments
    │   │   ├── train_eval.py                       # train and evaluate list-cvae model
    │   │   ├── evaluate.py                         # compare models
    │   │   ├── trainers.py                         # handlers with train, validate cycles
    │   │   └── __init__.py
    │   └── utils
    │       └── ...
    ├── configs
    │   ├── ifunny_test_config.yml
    │   └── movielens10m_config.yml
    ├── notebooks
    │   └── ...
    ├── data
    │   └── ...  
    └── README.md

## Experiment run

### Experiment: movielens10M
    cd /srv/ml_content_rate/experiments/DS-223-conditional-cvae-model
    wget https://files.grouplens.org/datasets/movielens/ml-10m.zip
    unzip ml-10m.zip
    export PYTHONPATH=${PYTHONPATH}:/srv/ml_content_rate/experiments/DS-223-conditional-cvae-model
    /usr/bin/python3.8 -u src/pipelines/prepare_data.py --config configs/movielens10m_config.yml
    /usr/bin/python3.8 -u src/pipelines/evaluate.py --config configs/movielens10m_config.yml
