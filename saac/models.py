from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import cv2 as cv
from sklearn import base


class SkinColorExtractor(ABC):
    def __init__(self,
                 lower_quantile: float = 0.5,
                 upper_quantile: float = 0.9,
                 rgb_threshold: float = True
                 ) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.rgb_threshold = rgb_threshold

    def luminance_mask(self, image: np.ndarray
                       ) -> Tuple[Tuple[int, int, int], np.ndarray]:
        img_l = cv.cvtColor(image, cv.COLOR_BGR2LAB)[:, :, 0]
        l_lower, l_upper = np.quantile(img_l, [self.lower_quantile, self.upper_quantile])

        mask = ((img_l >= l_lower) & (img_l <= l_upper))
        return mask

    @staticmethod
    def rgb_mask(image: np.ndarray
                 ) -> Tuple[Tuple[int, int, int], np.ndarray]:
        mask = (image[:, :, 0] < image[:, :, 2]) & (image[:, :, 1] < image[:, :, 2])
        return mask

    @abstractmethod
    def extract(self, image: np.ndarray
                ) -> Tuple[Tuple[int, int, int], np.ndarray]:
        return


class SkinColorMeanExtractor(SkinColorExtractor):
    def __init__(self,
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)

    def extract(self, image: np.ndarray
                ) -> Tuple[Tuple[int, int, int], np.ndarray]:
        mask = self.luminance_mask(image)
        if self.rgb_threshold:
            mask = mask & self.rgb_mask(image)
        blue, green, red = image[mask].mean(axis=0)
        return (red, green, blue), mask


class SkinColorModeExtractor(SkinColorExtractor):
    def __init__(self,
                 hist_bins: int = 20,
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)
        self.hist_bins = hist_bins

    def extract(self, image: np.ndarray
                ) -> Tuple[Tuple[int, int, int], np.ndarray]:
        mask = self.luminance_mask(image)
        if self.rgb_threshold:
            mask = mask & self.rgb_mask(image)
        hist, edges = np.histogramdd(image[mask], bins=self.hist_bins)
        mode_idx = np.array(np.unravel_index(np.argmax(hist, axis=None), hist.shape))
        blue, green, red = [np.take(edges[c], np.array((i, i + 1)), mode='clip').mean()
                            for c, i in enumerate(mode_idx)]
        return (red, green, blue), mask


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
