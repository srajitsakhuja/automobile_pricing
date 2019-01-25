import pandas as pd


class DataManipulator:
    def __init__(self, dataset, replacement_dict, numeric_coersion):
        self.dataset = dataset
        self.make_replacements(replacement_dict)
        self.make_numeric(numeric_coersion)

    def make_replacements(self, replacement_dict):
        for x, y in replacement_dict.items():
            self.dataset = self.dataset.replace(x, y)

    def make_numeric(self, numeric_coersion):
        for column in numeric_coersion:
            self.dataset[column] = pd.to_numeric(self.dataset[column])
