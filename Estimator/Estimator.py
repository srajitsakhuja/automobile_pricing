from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


class Estimator:
    def __init__(self, dataset, label):
        self.dataset = dataset
        self.label = label
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None
        self.classifier = None
        self.train_test_split()
        self.fit()
        self.calculate_metrics(self.x_test, self.y_test)

    def train_test_split(self):
        y = self.dataset[self.label]
        x = self.dataset.drop(self.label, axis=1)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=.25, random_state=42)

    def fit(self):
        self.classifier = SVR(gamma='scale', C=1.0, epsilon=0.2)
        self.classifier = RandomForestRegressor(n_estimators=13)
        self.classifier = LinearRegression()
        self.classifier.fit(self.x_train, self.y_train)

    def calculate_metrics(self, x, y):
        prediction = self.classifier.predict(x)
        actual = y
        self.mean_absolute_error(prediction, actual)
        self.mean_squared_error(prediction, actual)
        self.score(x, y)

    @staticmethod
    def mean_absolute_error(prediction, actual):
        print(f'MEAN ABS ERROR:{(abs(prediction - actual).mean())}')

    @staticmethod
    def mean_squared_error(prediction, actual):
        print(f'MEAN SQUARED ERROR:{((prediction - actual)**2).mean()}')

    def score(self, x, y):
        print(self.classifier.score(x, y))
