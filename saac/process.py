import cv2 as cv
import numpy as np
from typing import Tuple, List, Optional, Union

from deepface import DeepFace

from saac.utils import quadrant_bboxes, crop_bbox


class ImageEqualizer:
    def __init__(self,
                clipLimit: float = 2.0,
                grid_size: int = 8
                ) -> None:
        self.clipLimit = clipLimit
        self.grid_size = grid_size

    def equalize(self,
                 image: np.ndarray,
                 clipLimit: float = 2.0,
                 grid_size: int = 8
                 ) -> np.ndarray:
        img_lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
        img_l, img_a, img_b = cv.split(img_lab)

        clahe = cv.createCLAHE(
            clipLimit=self.clipLimit,
            tileGridSize=(self.grid_size, self.grid_size)
        )
        img_l = clahe.apply(img_l)

        img_lab = cv.merge((img_l, img_a, img_b))
        img_cl = cv.cvtColor(img_lab, cv.COLOR_LAB2BGR)
        return img_cl


class SkinColorExtractor:
    def __init__(self,
                lower_quantile: float = 0.4,
                upper_quantile: float = 0.9
                ) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def extract(self, image: np.ndarray
                ) -> Tuple[Tuple[int, int, int], np.ndarray]:
        img_l = cv.cvtColor(image, cv.COLOR_BGR2LAB)[:, :, 0]
        l_lower, l_upper = np.quantile(img_l, [self.lower_quantile, self.upper_quantile])

        mask = ((img_l >= l_lower) & (img_l <= l_upper))
        blue, green, red = image[mask].mean(axis=0)

        return (red, green, blue), mask


def df_bbox(region) -> List[int]:
    bbox = [
        region['x'],
        region['y'],
        region['x'] + region['w'],
        region['y'] + region['h']
    ]
    return bbox


def df_predictions(analysis: List[int], actions: Tuple[int]) -> dict:
    predictions = {action: analysis[action] for action in actions}
    predictions['bbox'] = df_bbox(analysis['region'])
    return predictions


DF_ACTIONS = (
    'age',
    'gender',
    'race',
    'emotion'
)


ALL_ACTIONS = DF_ACTIONS + ('skin',)


class MidJourneyProcessor:
    def __init__(self):
        pass

    @staticmethod
    def image_predictions(
            image: np.ndarray,
            actions: Tuple[str] = ('gender', 'skin'),
            models: Optional[dict] = None,
            detector_backend: str = 'mtcnn',
            min_size: int = 20,
            equalizer: Optional[Union[bool, ImageEqualizer]] = None,
            extractor: Optional[SkinColorExtractor] = None
    ) -> Optional[dict]:

        assert (all(action in ALL_ACTIONS for action in actions))

        if equalizer and type(equalizer) == bool:
            equalizer = ImageEqualizer()

        if 'skin' in actions and extractor is None:
            extractor = SkinColorExtractor()

        if equalizer:
            image = equalizer.equalize(image)

        predictions = None
        try:
            r = DeepFace.analyze(
                img_path=image,
                actions=(a for a in actions if a in DF_ACTIONS),
                models=models,
                detector_backend=detector_backend,
                enforce_detection=True,
                prog_bar=False
            )
            if r['region']['w'] >= min_size and r['region']['h'] >= min_size:
                predictions = df_predictions(r, (a for a in actions if a in DF_ACTIONS))

                if 'skin' in actions:
                    color, _ = extractor.extract(crop_bbox(image, predictions['bbox']))
                    predictions['skin color'] = color
        except ValueError:
            pass

        return predictions

    @staticmethod
    def quadrant_predictions(
            image: np.ndarray,
            **kwargs: dict,
    ) -> List[Optional[dict]]:

        results = []
        for bbox in quadrant_bboxes(image.shape[:2]):
            img_quadrant = crop_bbox(image, bbox)
            predictions = MidJourneyProcessor.image_predictions(
                img_quadrant,
                **kwargs
            )
            if predictions:
                predictions['bbox'] = [x + y for x, y in zip(predictions['bbox'], bbox[:2] * 2)]
            results.append(predictions)
        return results
