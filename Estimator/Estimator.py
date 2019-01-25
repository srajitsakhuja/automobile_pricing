from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


class Estimator:
    def __init__(self, dataset, label):
        self.dataset = dataset
        self.label = label
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None
        self.classifier = None
        self.train_test_split()
        self.fit()
        self.score(self.x_test, self.y_test)

    def train_test_split(self):
        y = self.dataset[self.label]
        x = self.dataset.drop(self.label, axis=1)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=.25, random_state=42)

    def fit(self):
        self.classifier = SVR(gamma='scale', C=1.0, epsilon=0.2)
        self.classifier = RandomForestRegressor(n_estimators=13)
        self.classifier = LinearRegression()
        self.classifier.fit(self.x_train, self.y_train)

    def score(self, x, y):
        print(self.classifier.score(x, y))
        print(mean_squared_error(y, self.classifier.predict(x)))
        print(mean_absolute_error(y, self.classifier.predict(x)))
