# SAAC
## Prompt Generation

## Images

### Gender Detection
As part of the bias audit, we tested and explored different models and techniques to best classify the gender of
labeled images. We settled upon using Deepface[^1] for gender detection - more specifically Deepface's facial attribute
analysis module.

#### Facial Attribute Analysis Module
Deepface's facial attribute analysis module which provides age, gender, facial expression and race predictions for a given
image. The module contained various parameters that could be adjusted for a given use case, we changed some of the module 
parameters for gender detection. By default, the module provides a dictionary output, as shown below.

```
{
			"region": {'x': 230, 'y': 120, 'w': 36, 'h': 45},
			"age": 28.66,
			"dominant_gender": "Woman",
			"gender": {
				'Woman': 99.99407529830933,
				'Man': 0.005928758764639497,
			}
			"dominant_emotion": "neutral",
			"emotion": {
				'sad': 37.65260875225067,
				'angry': 0.15512987738475204,
				'surprise': 0.0022171278033056296,
				'fear': 1.2489334680140018,
				'happy': 4.609785228967667,
				'disgust': 9.698561953541684e-07,
				'neutral': 56.33133053779602
			}
			"dominant_race": "white",
			"race": {
				'indian': 0.5480832420289516,
				'asian': 0.7830780930817127,
				'latino hispanic': 2.0677512511610985,
				'black': 0.06337375962175429,
				'middle eastern': 3.088453598320484,
				'white': 93.44925880432129
			}
		}
```

#### Parameters

##### Actions
By default, the module will generate an output for the following actions: age, gender, facial expression, and race.
For our use case, we used the gender action and omitted the use of the age, facial expression, and race actions. In doing so,
the output of the module would only produce the probability in which a face is a woman or a man and the dominant gender of the face.

##### Models
The model used for gender prediction was trained by the author of Deepface on the [VGG-Face](https://www.robots.ox.ac.uk/~vgg/software/vgg_face/) 
structure. Pre-trained weights for the model are saved and called upon whenever any gender predictions are made. The module
allows for users to provide their own pre-trained models; however, we decided to use the default gender model weights trained by and provided by the author.

##### Face detection
The module uses a face detector prior to identifying gender - the face detector draws a tight bound around the face of the image entity. 
The default face detector for the module is opencv. Other face detection options include retinaface, mtcnn, ssd or dlib. After examining all possible face detection options, we used the MTCNN face detector. MTCNN was chosen over opencv and other face detection options because it generally had a accuracy ~1-2% greater than that of all other options.

#### Calibration

##### Process
Upon initial examination, the Deepface predictions seemed to skew towards mislabeling women as men as shown in Figure 1. As such,
we explored ways to mitigate this bias through calibrating the gender detection classifier. For calibration, we pursued a
cross-validation approach through using [CalibratedClassifierCV](https://scikit-learn.org/stable/modules/calibration.html).

##### Results Comparison
The calibration of the gender classifier evidently mitigated the bias that we were seeing with the uncalibrated model. As shown in Figure 6, the calibration plot no longer skews towards one particular class. With the uncalibrated model, 148 of the images of women were mislabeled, whereas only 5 of the images of men were mislabeled - as shown in Figure 3. Post-calibration, only 14 of the images of the women were mislabeled, whereas 20 of the images of men were mislabeled - as shown in Figure 6. Calibrating the model lessened the overwhelming mislabeling of women as men. Calibration of the gender detection model also produced an increase in accuracy by 6% as the accuracy went from 82% to 88% as shown in Figure 2 & 5.



<p align="center">
<img width="290" alt="Uncalibrated_Results" src="https://media.github.trssllc.com/user/146/files/b543f640-ed1e-4dbc-8c2d-710216b6f33a" style="width:50%">
</p>
<p align = "center">
<i>Fig. 1</i> - Uncalibrated Model - Calibration Plot
</p>

<p align="center">
<img width="213" alt="Uncalibrated_Metrics" src="https://media.github.trssllc.com/user/146/files/61bee301-f7ef-45e6-b705-7f7bc096ee6e" style="width:50%">
</p>
<p align = "center">
<i>Fig. 2</i> - Uncalibrated Metrics
</p>


<p align="center">
<img width="289" alt="Gender_Detection_Matrix" src="https://media.github.trssllc.com/user/146/files/3cc2630e-0177-441a-b4bc-088a694fae52" style="width:50%">
</p>
<p align = "center">
<i>Fig. 3</i> - Uncalibrated Model Results
</p>


		   
<p align="center">
<img width="291" alt="Calibrated_Results" src="https://media.github.trssllc.com/user/146/files/3cae1ba3-359f-481c-b3da-598c986b23db" style="width:50%">
</p>
<p align = "center">
<i>Fig. 4</i> - Calibrated Model - Calibration Plot
</p>


<p align="center">
<img width="209" alt="Calibrated_Metrics" src="https://media.github.trssllc.com/user/146/files/80aa7146-3d73-4188-85c3-6d32b49c1b0b" style="width:50%">
</p>
<p align = "center">
<i>Fig. 5</i> - Calibrated Metrics
</p>

<p align="center">
<img width="277" alt="Calibrated_Gender_Detection_Matrix" src="https://media.github.trssllc.com/user/146/files/50ccba69-a546-4c8f-b892-6a3f9178d0ea" style="width:50%">
</p>
<p align = "center">
<i>Fig. 6</i> - Calibrated Model Results
</p>


#### Limitations & biases
Some limitations and biases with Deepface include:
- The model was trained on a labeled dataset of ~4 million faces belonging to over 4,000 individuals. Due to the training data
  being solely comprised of photorealistic images, there are limitations in how the model predicts the gender of images that are more abstract
  in nature.
- Results are skewed towards one gender. We found that generally Deepface had a tendency to mislabel women as men (an issue that is addressed using calibration).


*****

### Skin Color Extraction
In addition to our work with gender detection, we explored techniques to best extract accurate skin tone information
from images of faces. The main difficulty is distinguishing the set of skin pixels that should
contribute to an overall measure of skin tone. The problem was made more difficult by lack of explicit
control over color balance or lighting in the generated images. Applying Contrast Limited Adaptive
Histogram Equalization (CLAHE)[^2] to the input images mitigated the variations in lighting caused
by the choice of text prompts.

To generate the skin tone measure, we followed the methodology below proposed by Harville et al.[^4]
1. Face detection
2. Skin pixel identification
3. Skin tone estimation

#### Face detection

For face detection, we used the same MTCNN face detector[^3] applied in the gender detection.
Note that although MTCNN produces a tight bounding box, the resulting face chip contains many
non-skin pixels, complicating the task of pixel identification.

#### Skin Pixel Identification

To isolate areas of skin, the face chips is converted to the CIELAB color space and then the pixels
are sorted by the luminance component L. Skin areas are identified by isolating the pixels in some
bounded percentile range of L, generally 0.5 to 0.9. The upper bound exclude specularities on the face
while the lower bound removes dark areas such as hair, nostrils, mouth and shadows.

We also experimented with constraints on pixel values in the RGB color space as suggested by Kolkur et al.[^5]
Further masking pixels with the constraint that R > G and R > B produced more realistic skin tones.

#### Skin Tone Estimation

Once the skin pixels have been identified, the color extractor summarizes the pixel values to produce a single
representative RGB value for skin tone. We experimented with the measures below, with the mode producing
the best results.

- Mean value - return the separate means of the RBG components of all skin pixels.
- Mode value - return the most frequent RGB skin pixel value as identified by a multi-dimension histogram.

### Results of image evaluation workflow 
Upon going through the image evaluation workflow, the resulting output CSVs include a CSV with uncalibrated Deepface predictions and a CSV with calibrated Deepface predictions. Each CSV contains information about the following:
| Column Name     | Value Description |
| ----------- | ----------- |
| image      | image name/path       |
| label   | 0 indicating that the image is of a woman, 1 indicating that the image is of a man        |
| bbox   | contains the bounding box coordinates of the face detection (Ex: {'x': 230, 'y': 120, 'w': 36, 'h': 45})        |
| gender.Woman   | probability that the image is a women        |
| gender.Man   | probability that the image is a man|

## Evaluation of Results

[^1]: Serengil, Sefik & Ozpinar, Alper. (2020). LightFace: A Hybrid Deep Face Recognition Framework. 10.1109/ASYU50717.2020.9259802.

[^2]: Pizer, Stephen & Amburn, E. & Austin, John & Cromartie, Robert & Geselowitz, Ari & Greer, Thomas & ter Haar Romeny, Bart & Zimmerman, John & Zuiderveld, Karel. (1987). Adaptive Histogram Equalization and Its Variations. Computer Vision, Graphics, and Image Processing. 39. 355-368. 10.1016/S0734-189X(87)80186-X. 

[^3]: Zhang, Kaipeng & Zhang, Zhanpeng & Li, Zhifeng & Qiao, Yu. (2016). Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks. IEEE Signal Processing Letters. 23. 10.1109/LSP.2016.2603342.

[^4]: Harville, Michael & Baker, Harlyn & Susstrunk, S.. (2005). Image-based measurement and classification of skin color. Proc IEEE Int Conf Image Process. 2. II - 374. 10.1109/ICIP.2005.1530070.

[^5]: Kolkur, S. & Kalbande, Dhananjay & Shimpi, P. & Bapat, C. & Jatakia, Janvi. (2017). Human Skin Detection Using RGB, HSV and YCbCr Color Models. 10.2991/iccasp-16.2017.51.

