import os
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import torch
from facenet_pytorch import MTCNN

from matplotlib import pyplot as plt


def draw_bboxes(image: Image, bboxes: np.array, probabilities: np.array):
	draw = ImageDraw.Draw(image)
	for bbox, prob in zip(bboxes, probabilities):
		draw.rectangle(bbox, fill=None, outline=None, width=1)
		draw.text(bbox[:2], f'{prob: .6f}', anchor='lb')


def display_bbox(midjourney_pngs, probability_results):
	i = 0
	for png, result in zip(midjourney_pngs, probability_results):
		img = Image.open(png)
		if result[0] is not None and result[1] is not None:
			draw_bboxes(img, *result)
			img.show()
			i += 1


def crop_bboxes(image: Image, bboxes: np.array):
	chips = list(map(lambda bbox: image.crop(bbox), bboxes))
	return chips


def process_directory(dir='./midjourney_zs/test'):
	midjourney_root = Path(dir)
	midjourney_pngs = list(midjourney_root.glob('*.png'))
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print('Running on device: {}'.format(device))
	mtcnn = MTCNN(
		image_size=160, margin=0, min_face_size=20,
		thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
		device=device
		)
	results = []

	for png_path in tqdm(midjourney_pngs):
		img = Image.open(png_path)
		result = mtcnn.detect(img)
		results.append(result)

	for idx in range(len(midjourney_pngs)):
		fp = midjourney_pngs[idx]
		fn = os.path.split(fp)[-1]
		fn = os.path.splitext(fn)[0]
		img = Image.open(fp)
		bboxes = results[idx][0]
		if bboxes is not None:
			chips = crop_bboxes(img, bboxes)

			for i, chip in enumerate(chips):
				#chip.show()
				chip.save(os.path.join(dir,'chipped',f'{fn}_{i}.png'))
		else:
			img.save(os.path.join(dir,'chipped',f'{fn}_{0}.png'))
if __name__=='__main__':
	process_directory()