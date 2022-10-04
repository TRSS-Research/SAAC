import numpy as np
from sklearn import base


class IdentityClassifier(
    base.TransformerMixin,
    base.ClassifierMixin,
    base.BaseEstimator
):
    def __init__(
            self,
            threshold
    ):
        super().__init__()
        self.classes_ = np.array(range(0, 2))
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        return np.stack((1 - X, X), axis=1)

    def predict(self, X):
        return np.array([1 if x >= self.threshold else 0 for x in X])


class CalibratedGenderModel:
    def __init__(self,
                 gender_model,
                 calibrator
                 ):
        self.gender_model = gender_model
        self.calibrator = calibrator

    def predict(self, x):
        df_predictions = self.gender_model.predict(x)[:, 1]
        calibrated = self.calibrator.predict_proba(df_predictions)
        return calibrated
