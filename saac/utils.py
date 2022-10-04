from typing import Tuple, List, Optional

import cv2 as cv
import matplotlib
import numpy as np
from matplotlib import pyplot as plt


def cv_imshow(
        image: np.ndarray,
        ax=None
) -> None:
    if ax:
        ax.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        ax.axis('off')
    else:
        plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        plt.axis('off')


def color_show(
        color: Tuple[int, int, int],
        size=1, ax=None
) -> None:
    color = tuple(c / 256 for c in color)
    rect = matplotlib.patches.Rectangle((0, 0), size, size, color=color)
    if ax is None:
        fig = plt.figure(figsize=(size, size))
        ax = fig.add_subplot(111)
    ax.add_patch(rect)
    ax.axis('off')


def quadrant_bboxes(
        img_size: Tuple[int, int]
) -> List[List[int]]:
    width, height = [d // 2 for d in img_size]
    bboxes = [[x0, y0, x0 + width, y0 + height] for x0 in [0, width] for y0 in [0, height]]
    return bboxes


def crop_bbox(image: np.ndarray, bbox: np.array) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    return image[y0:y1, x0:x1]


def draw_bbox(
        image: np.ndarray,
        bbox: List[int],
        text: Optional[str] = None,
        color: Tuple[int, int, int] = (255,) * 3,
        thickness: int = 1,
        fontFace: int = cv.FONT_HERSHEY_SIMPLEX,
        fontScale: float = 0.5
) -> np.ndarray:
    cv.rectangle(
        image,
        bbox[:2],
        bbox[2:],
        color=color,
        thickness=thickness
    )

    if text:
        (_, h), _ = cv.getTextSize(
            text,
            fontFace=fontFace,
            fontScale=fontScale,
            thickness=thickness
        )
        cv.putText(
            image,
            text,
            (bbox[0], bbox[1] + h),
            fontFace=cv.FONT_HERSHEY_SIMPLEX,
            fontScale=fontScale,
            color=color,
            thickness=thickness
        )

    return image
