from pathlib import Path
from typing import Tuple, List, Optional, Union

import cv2 as cv
import deepface.detectors.FaceDetector
import joblib
import numpy as np
from deepface import DeepFace
import pandas as pd
from tqdm import tqdm

from saac.models import CalibratedGenderModel, SkinColorExtractor, SkinColorMeanExtractor, SkinColorModeExtractor
from saac.utils import quadrant_bboxes, crop_bbox
import os
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
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


def df_bbox(region) -> List[int]:
    bbox = [
        region['x'],
        region['y'],
        region['x'] + region['w'],
        region['y'] + region['h']
        ]
    return bbox


def df_predictions(analysis: dict[str], actions: Tuple[str]) -> dict:
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
    def batch_predictions(
            image_regions : List[Tuple[np.ndarray,Tuple[float,float]]],
            actions : Tuple[str] = ('gender','skin'),
            detector_backend: str = 'mtcnn',
            min_size: int = 20,
            models: Optional[dict] = None,
            equalizer: Optional[Union[bool, ImageEqualizer]] = None,
            extractor: Optional[SkinColorExtractor] = None
            ) -> List[dict]:

        assert (all(action in ALL_ACTIONS for action in actions))
        retVal = []
        if equalizer and type(equalizer) == bool:
            equalizer = ImageEqualizer()

        if 'skin' in actions and extractor is None:
            extractor = SkinColorModeExtractor()
        batch = []
        # regions = []
        for image,region in image_regions:
            x,y,w,h = region
            if w >= min_size and h >= min_size:
                if equalizer:
                    batch.append(equalizer.equalize(image))
                else:
                    batch.append(image)
                # regions.append(region)
        predictions = None
        try:
            r = DeepFace.analyze(
                img_path=batch,
                actions=tuple(a for a in actions if a in DF_ACTIONS),
                models=models,
                detector_backend=detector_backend,
                enforce_detection=True,
                prog_bar=False
                )
            i = 0
            for k in r.keys():
                predictions = df_predictions(r[k], tuple(a for a in actions if a in DF_ACTIONS))
                if 'skin' in actions:
                    color, _ = extractor.extract(crop_bbox(batch[i], predictions['bbox']))
                    predictions['skin color'] = color
                retVal.append(predictions)
                i+=1
        except ValueError:
            pass
        return retVal

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
            extractor = SkinColorModeExtractor()

        if equalizer:
            image = equalizer.equalize(image)

        predictions = None
        try:
            r = DeepFace.analyze(
                img_path=image,
                actions=tuple([a for a in actions if a in DF_ACTIONS]),
                models=models,
                detector_backend=detector_backend,
                enforce_detection=True,
                prog_bar=False
                )
            if r['region']['w'] >= min_size and r['region']['h'] >= min_size:
                predictions = df_predictions(r, tuple(a for a in actions if a in DF_ACTIONS))

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


def process_multiple(raw_root):
    df_default_models = {
        'age': DeepFace.build_model('Age'),
        'gender': DeepFace.build_model('Gender'),
        'emotion': DeepFace.build_model('Emotion'),
        'race': DeepFace.build_model('Race'),
        'detector': deepface.detectors.FaceDetector.build_model('mtcnn')
        }

    calibrated_model_path = Path(os.path.join(ANALYSIS_DIR,'models','gender_model','gender_model_default_calibrated.joblib'))
    calibrated_clf = joblib.load(calibrated_model_path)

    calibrated_gender_model = CalibratedGenderModel(
        gender_model=df_default_models['gender'],
        calibrator=calibrated_clf
        )

    # gender_model = df_default_models['gender']
    gender_model = calibrated_gender_model

    color_extractor = SkinColorMeanExtractor()

    kwargs = {
        'equalizer': True,
        'actions': ('gender', 'skin'),
        'models': {'gender': gender_model},
        'extractor': color_extractor,

        }

    def prompt_extract(path: str) -> str:
        return ' '.join(path.split('_')[1:-1])

    all_predictions = []

    for quad_path in tqdm(list(raw_root.glob('*.png'))):
        quad_image = cv.imread(str(quad_path))
        prompt = prompt_extract(quad_path.stem)
        # list of (aligned_face, region)
        faces_regions = deepface.detectors.FaceDetector.detect_faces(detector_backend='mtcnn',
                                                                     face_detector=df_default_models['detector'],img=quad_image)

        faces_regions = faces_regions if isinstance(faces_regions,list) else [faces_regions]

        predictions = MidJourneyProcessor.batch_predictions(faces_regions,
                                                            **kwargs
                                                            )
        if predictions is None:
            print('None', faces_regions[1])
        else:
            for idx, pred in enumerate(predictions):
                #print(pred)
                pred['image'] = quad_path.stem
                pred['quadrant'] = idx
                pred['prompt'] = prompt
            all_predictions.extend(predictions)
    results_df = pd.json_normalize(all_predictions)

    lead_cols = [
        'prompt',
        'image',
        'quadrant',
        'bbox'
        ]

    results_df = results_df.reindex(columns=lead_cols + [col for col in results_df.columns if col not in lead_cols])
    results_df.to_csv(Path('./data/midjourney_deepface_calibrated_equalized.csv'), index=False)
    return results_df

def process_images(raw_images_dir: str = './data/mj_raw'):

    raw_root = Path(raw_images_dir)

    return process_multiple(raw_root=raw_root)


if __name__ == '__main__':
    process_images()
