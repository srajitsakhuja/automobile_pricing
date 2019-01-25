import matplotlib.pyplot as plt
import numpy as np
import math


class DataAnalyzer:
    def __init__(self, dataset, print_flag=True, graph_flag=True):
        self.dataset = dataset
        self.print_flag = print_flag
        self.non_numeric_columns = None
        self.numeric_columns = None

        self.dataset_dtypes()
        self.dataset_stats()
        self.dataset_missing_counts()

        if self.print_flag:
            print(f'SHAPE:{self.dataset.shape}')
            print("*"*150)
        if graph_flag:
            self.dataset_histogram()
            self.dataset_barcharts()

    def dataset_dtypes(self):
        self.non_numeric_columns = self.dataset.dtypes == np.object
        self.numeric_columns = self.dataset.dtypes != np.object
        self.numeric_columns = list(self.numeric_columns[self.numeric_columns].index.values)
        self.non_numeric_columns = list(self.non_numeric_columns[self.non_numeric_columns].index.values)
        if self.print_flag:
            print("\nDATA TYPES:")
            print(self.dataset.dtypes)
            print(f'Non-numeric Columns{self.non_numeric_columns}')
            print(f'Numeric Columns{self.numeric_columns}')

    def dataset_stats(self):
        if self.print_flag:
            print("\nSTATISTICS:")
            print(self.dataset.describe().transpose())

    def dataset_missing_counts(self):
        missing_counts = (self.dataset.isnull()).sum() + (self.dataset == '?').sum()
        if self.print_flag:
            print("\nMISSING COUNTS:")
            print(missing_counts[missing_counts != 0])

    def non_numeric_value_counts(self):
        if self.print_flag:
            for column in self.non_numeric_columns:
                print(f'{column}\n{self.dataset[column].value_counts()}\n\n')

    def dataset_histogram(self):
        self.dataset[self.numeric_columns].hist(figsize=[25, 25])
        if self.print_flag:
            plt.show()

    def dataset_barcharts(self, column_names=[]):
        if len(column_names) == 0:
            column_names = self.non_numeric_columns
        fig = plt.figure(figsize=[25, 25])
        plt.subplots_adjust(wspace=0.12, hspace=1, left=.025, bottom=.15, top=.8, right=1)
        data_with_nans = self.dataset.replace(np.nan, '?')  # plotting with np.nan is not possible
        for i in range(1, len(column_names)+1):
            rows = int(math.sqrt(len(column_names)))
            fig.add_subplot(rows, int(len(column_names)/rows)+1, i)
            plt.bar(data_with_nans[column_names[i-1]].value_counts().index.values,
                    data_with_nans[column_names[i-1]].value_counts().values)
            plt.xticks(rotation=90)
            plt.title(column_names[i-1])
        if self.print_flag:
            plt.show()
