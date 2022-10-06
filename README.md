# SAAC
## Prompt Generation

## Images
### Face Detection - 
- Model used (deep face) - link to model 
- What the default parameters 
- What additional  parameters exist (age, race, emotion) >untested not defaults

### Gender Detection
- Model used (deep face)
- Calibration process / data used / 
- Calibration results vs Original Results
- Possible Limitations of the model , skew of results - bias towards any one gender

### Skin Color Extraction
As part of the bias audit, we explored techniques to best extract accurate skin tone information
from images of faces. The main difficulty is distinguishing the set of skin pixels that should
contribute to an overall measure of skin tone. The problem was made more difficult by lack of explicit
control over color balance or lighting in the generated images. Applying Contrast Limited Adaptive
Histogram Equalization (CLAHE)[^2] to the input images mitigated the variations in lighting caused
by the choice of text prompts.

To generate the skin tone measure, we followed the methodolgy below proposed by Harville et al.[^4]
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
while the lower bound removes dark areas sucha as hair, nostrils, mouth and shadows.

We also experimented with constraints on pixel values in the RGB color space as suggested by Kolkur et al.[^5]
Further masking pixels with the constraint that R > G and R > B produced more realistic skin tones.

#### Skin Tone Estimation

Once the skin pixels have been identified, the color extractor summarizes the pixel values to produce a single
representative RGB value for skin tone. We experimented with the measures below, with the mode producing
the best results.

- Mean value - return the separate means of the RBG compoents of all skin pixels.
- Mode value - return the most frequent RGB skin pixel value as identified by a multi-dimension histrogram.

### Results of image evaluation workflow 
- Description of output csv

## Evaluation of Results

[^1]: Serengil, Sefik & Ozpinar, Alper. (2020). LightFace: A Hybrid Deep Face Recognition Framework. 10.1109/ASYU50717.2020.9259802.

[^2]: Pizer, Stephen & Amburn, E. & Austin, John & Cromartie, Robert & Geselowitz, Ari & Greer, Thomas & ter Haar Romeny, Bart & Zimmerman, John & Zuiderveld, Karel. (1987). Adaptive Histogram Equalization and Its Variations. Computer Vision, Graphics, and Image Processing. 39. 355-368. 10.1016/S0734-189X(87)80186-X. 

[^3]: Zhang, Kaipeng & Zhang, Zhanpeng & Li, Zhifeng & Qiao, Yu. (2016). Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks. IEEE Signal Processing Letters. 23. 10.1109/LSP.2016.2603342.

[^4]: Harville, Michael & Baker, Harlyn & Susstrunk, S.. (2005). Image-based measurement and classification of skin color. Proc IEEE Int Conf Image Process. 2. II - 374. 10.1109/ICIP.2005.1530070.

[^5]: Kolkur, S. & Kalbande, Dhananjay & Shimpi, P. & Bapat, C. & Jatakia, Janvi. (2017). Human Skin Detection Using RGB, HSV and YCbCr Color Models. 10.2991/iccasp-16.2017.51.
