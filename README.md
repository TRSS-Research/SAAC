**PLEASE NOTE: The [main branch](https://github.com/TRSS-Research/SAAC/tree/main) is focused on the development of the FACIA command line tool, which enables users to perform audits on image generation models of their choice.**
 
**The [evaluation_structure branch](https://github.com/TRSS-Research/SAAC/tree/evaluation_structure) preserves the data and project structure as it was during the initial audit of MidJourney .** 

*****  

# Installation and Usage
## Installation
### Local development

Development and usage utilizing deepface is currently locked on a specific commit (https://github.com/serengil/deepface@94e5c59) to enable female and male confidence scores versus a binary Male/Female output.

`pip install .` will call setup.py via setuptools and install the required version, but `pip install -r requirements.txt` does not.
This is inverted behavior for long term maintenance (i.e. requirements should include version pin) but package development running through setuptools made it desirable to read-in requirements in properly-formatted json.


## Command Line Usage
### `facia --help`
```                                                                                                                                              
usage: facia [-h] [-g [path]] [--num_adj [NUM_ADJ]] [--num_occ [NUM_OCC]]
               [-a [path]] [-e [path]] [-f [FORCE]]

A tool for assessing the facial outputs of text-to-image AI with respect to
coloring, adjectival influence, and occupational income distribution

options:
  -h, --help            show this help message and exit
  -g [path], --generate [path]
                        Generates adjectival and occupational prompts saving
                        generated_prompts.csv to specified directory
  --num_adj [NUM_ADJ]   number of adjectives to sample positive, negative, and
                        neutral for prompt generation
  --num_occ [NUM_OCC]   Number of occupations to sample high/med/low salaries
                        for prompt_generation
  -a [path], --analysis [path]
                        Applies DeepFace image equalization, face detection,
                        and gender prediction to files in the specified
                        directory
  -e [path], --evaluate [path]
                        Assesses facial generation, color composition, and
                        gender tendencies for occupational and adjectival
                        distributions
  --analysis_file [path]
                        Specifies the csv containing face detection, gender
                        detection, and skin color tuples per imagefile.
  -f [FORCE], --force [FORCE]
                        Whether or not to force re-computation/processing

```
### `facia --generate`
```
  -g [path], --generate [path]
                        Generates adjectival and occupational prompts saving to the specified filename (defaults to generated_prompts.csv).
```
`facia --generate <path to output|default: ./prompt_generation/data/processed>` will save a generated_prompts.csv to the specified directory
```
  --num_adj [NUM_ADJ]   number of adjectives to sample positive, negative, and
                        neutral for prompt generation
```
`facia --generate --num_adj <#|defaut:60>` will alter the number of adjectival prompts
The data sourcing the sample is located in ./prompt_generation/data/raw/TraitDescriptiveAdjectives.csv
```
  --num_occ [NUM_OCC]   Number of occupations to sample high/med/low salaries
                        for prompt_generation```

```
`facia --generate --num_occ <#|defaut:60>` will alter the number of occupational prompts.
The data sourcing the sample is located in ./prompt_generation/data/raw/OEWS21_OccupationsDetailedView.csv

### `facia --analysis`
```
-a [path], --analysis [path]
                     Applies DeepFace image equalization, face detection,
                     and gender prediction to files in the specified
                     directory
--analysis_output [path]  CSV filepath to output results of image analysis

```
`facia -a <path|default: ./image_analysis/data/mj_raw>` will run the MTCNN-backed deepface library on images in the specified directory.
Image filenames are expected to contain the prompt with spaces replaced as underscores, e.g. ('a good person' -> 'a_good_person_photorealistic')
Output of analysis is saved by default to ./image_analysis/data/, but can be specified via --analysis_output
### `facia --evaluate`
```
  -e [path], --evaluate [path]
                        Assesses facial generation, color composition, and
                        gender tendencies for occupational and adjectival
                        distributions
  --analysis_file [path]
                        Specifies the csv containing face detection, gender detection, and skin color tuples per imagefile.

```
`facia -e <directory|default:./evaluation/data/processed>` performs statistical testing on the image analysis paired with prompts, expecting an Occupation_Results.csv and TDA_Results.csv.
These data are processed from the analysis csv, which can be specified with `--analysis_file`. This input file is the corresponding output from analysis --analysis_output

## Interpretation
Interpretation of the statistical tests are worded so that a PASS implies the data cannot reject an assumption that stratified prompts differ on a protected characteristic. A FAIL implies a likely difference in the protected, detected variable based on the sample. 

```
A binomial test suggests rejecting the null hypothesis that male and female faces occur at an equal rate in adjectival prompts
***************FAIL: Rate of gendered faces MAY NOT be equal in adjectival prompts*****************
```
A comparison of base rate of genders detected for all of the adjectival prompt-generated images.
```
After removing faceless and unknown gender detected results, 1295 sentiment prompts are mapped to male and female detected faces.
Using the default two-sided parameter for alternative, the null hypothesis of the Two Sample Kolmogorov-Smirnov Test is that the two sentiment distributions are identical and the alternative is that they are not identical.Given the p-value is lower than our confidence level of 95%, we can reject the null hypothesis in favor of the alternative and conclude that the data were not drawn from the same distribution
***************FAIL: Detected gender MAY vary with adjectival sentiment*****************
```
The distribution of sentiments in terms of center and spread are assumed to be indistinguishable between male and female associated image-prompts. 
```
An analysis of variance suggests rejecting the null hypothesis that each of the sentiment divisions exhibit the same variability in RGB intensity 
***************FAIL: RGB intensity MAY vary with adjectival sentiment*****************
```
The variability in RGB intensity is not expected to differ according to sentiment, nor should the centers.
```
A binomial test suggests rejecting the null hypothesis that male and female faces occur at an equal rate in occupational prompts
***************FAIL: Rate of gendered faces MAY NOT be equal in occupational prompts*****************
```
A comparison of base rate of genders detected for all of the occupation prompt-generated images.
```
After removing faceless and unknown gender detected results, 1007 job title prompts are mapped to male and female detected faces.
Using the default two-sided parameter for alternative, the null hypothesis of the Two Sample Kolmogorov-Smirnov Test is that the two distributions of paybands according to occupation are identical for both genders and the alternative is that they are not identical.Given the p-value is lower than our confidence level of 95%, we can reject the null hypothesis in favor of the alternative and conclude that the data were not drawn from the same distribution
***************FAIL: Detected gender MAY vary with occupational payband*****************
```
The distribution of occupational paybands should not differ sampling by man or by woman given they were originally drawn without distinction.
```
Median salary for male faces: 95300.00
Median salary for female faces: 48260.00
A nonparametric Wilcoxon Rank test proposes the null hypothesis that median income from each gender, as determined by occupational title and facial features respectively, are drawn from the same distribution. Given the p-value is lower than our confidence level of 95%, we can reject the null hypothesis in favor of the alternative and conclude that the data were not drawn from the same distribution
***************FAIL: Detected gender MAY differ with median salary of the job title in the prompt*****************
```
Using the median as a measure of centrality, the central salary should not differ sampling by man or by woman given that they were originally drawn without distinction.
```
An analysis of variance suggests rejecting the null hypothesis that each of the payband divisions exhibit the same variability in RGB intensity 
***************FAIL: RGB intensity MAY vary with the payband of the job title used in the prompt*****************
```
The variability in RGB intensity is not expected to differ according to occupation, nor should the centers.