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
As part of the audit, we considered techniques to best to extract accurate skin tone information from images of faces.
We used the same MTCNN face detector.
- Luminance mask - link to paper[^2]
- RBG constraints[^3]




### Results of image evaluation workflow 
- Description of output csv

## Evaluation of Results

[^1]: Serengil, Sefik & Ozpinar, Alper. (2020). LightFace: A Hybrid Deep Face Recognition Framework. 10.1109/ASYU50717.2020.9259802. 

[^2]: Zhang, Kaipeng & Zhang, Zhanpeng & Li, Zhifeng & Qiao, Yu. (2016). Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks. IEEE Signal Processing Letters. 23. 10.1109/LSP.2016.2603342. 

[^3]: Harville, Michael & Baker, Harlyn & Susstrunk, S.. (2005). Image-based measurement and classification of skin color. Proc IEEE Int Conf Image Process. 2. II - 374. 10.1109/ICIP.2005.1530070.

[^4]: Kolkur, S. & Kalbande, Dhananjay & Shimpi, P. & Bapat, C. & Jatakia, Janvi. (2017). Human Skin Detection Using RGB, HSV and YCbCr Color Models. 10.2991/iccasp-16.2017.51.
