import argparse
from DatasetManager.DatasetManager import DatasetManager
from DataAnalyzer.DataAnalyzer import DataAnalyzer
from DataManipulator.DataManipulator import DataManipulator
from FeatureManager.FeatureManager import FeatureManager
from Estimator.Estimator import Estimator
import numpy as np


ARGPARSER = argparse.ArgumentParser()
ARGPARSER.add_argument("dataset", help="enter path to dataset file", type=str)
ARGPARSER.add_argument("description", help="enter path to description file", type=str)
DATASET_FILE_PATH = ARGPARSER.parse_args().dataset
DESCRIPTION_FILE_PATH = ARGPARSER.parse_args().description
NUMERIC_COERSION = ['normalized-losses', 'bore', 'stroke', 'horsepower', 'peak-rpm', 'price']
REPLACEMENT_DICT = {'4wd': 'fwd', '?': np.nan}
NOMINAL = ['make', 'body-style', 'engine-type', 'fuel-system']
ORDINAL = ['num-of-cylinders']
DICHOTOMOUS = ['fuel-type', 'aspiration', 'num-of-doors', 'drive-wheels', 'engine-location']

if __name__ == "__main__":
    dataset_manager = DatasetManager(DATASET_FILE_PATH, DESCRIPTION_FILE_PATH)
    dataset = dataset_manager.dataset

    # data_analyzer = DataAnalyzer(dataset, print_flag=True, graph_flag=False)

    data_manipulator = DataManipulator(dataset, REPLACEMENT_DICT, NUMERIC_COERSION)
    dataset = data_manipulator.dataset

    data_analyzer = DataAnalyzer(dataset, print_flag=False, graph_flag=False)

    feature_manager = FeatureManager(dataset, data_analyzer.numeric_columns, data_analyzer.non_numeric_columns, NOMINAL, DICHOTOMOUS, ORDINAL)
    # dataset = feature_manager.dataset

    data_analyzer = DataAnalyzer(dataset, print_flag=False, graph_flag=False)

    estimator = Estimator(dataset, label='price')


