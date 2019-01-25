from sklearn.preprocessing import LabelEncoder


class FeatureManager:
    def __init__(self, dataset, numeric_features,  non_numeric_features, nominal_features, dichotomous_features, ordinal_features):
        self.dataset = dataset
        self.nominal = nominal_features
        self.dichotomous = dichotomous_features
        self.ordinal = ordinal_features
        self.non_numeric = non_numeric_features
        self.numeric = numeric_features
        self.fix_non_numeric_features()
        self.fix_missing_values()

    def fix_non_numeric_features(self):
        for feature in self.non_numeric:
            # print(feature)
            label_encoder = LabelEncoder()
            altered = label_encoder.fit_transform(self.dataset[feature].astype(str))
            self.dataset[feature+'_altered'] = altered
            self.dataset.drop(feature, inplace=True, axis=1)

    def fix_missing_values(self):
        # Uncomment while taking notes!
        # x1 = self.dataset['normalized-losses'][:25]
        # x2 = self.dataset.interpolate(method='quadratic', axis=0, limit_direction='both')['normalized-losses'][:25]
        # x3 = self.dataset.interpolate(method='quadratic', axis=0, limit_direction='both').interpolate(method='linear',
        # axis=0, limit_direction='both')['normalized-losses'][:25]
        # df = pd.DataFrame([x1, x2, x3]).transpose()
        # print(df)
        self.dataset.interpolate(method='quadratic', axis=0, inplace=True)
        self.dataset.interpolate(method='linear', axis=0, inplace=True, limit_direction='both')
