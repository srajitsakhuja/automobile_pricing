import pandas as pd


class DatasetManager:
    def __init__(self, dataset_file_path, description_file_path):
        self.dataset = self.read_dataset(dataset_file_path, self.read_description(description_file_path))

    @staticmethod
    def read_dataset(file_path, columns):
        return pd.read_csv(file_path, header=None, names=columns)

    @staticmethod
    def read_description(file_path):
        with open(file_path, 'r') as file:
            description = file.readlines()
        description = [line.strip() for line in description]
        return description
