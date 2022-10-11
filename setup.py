import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()
if 'deepface' in required:
    required.pop(required.index('deepface'))
    required.append("deepface @ git+https://github.com/serengil/deepface@94e5c59")
    print(required)

setuptools.setup(
    name="saac",
    version="0.0.1",
    author="TRSS",
    author_email="valeria.rozenbaum@trssllc.com",
    description="Framework for text-to-image auditing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.trssllc.com/datascience/SAAC",
    package_data={
        'saac':['./image_analysis/models/gender_model/gender_model_default_calibrated.joblib',
                './prompt_generation/data/raw/TraitDescriptiveAdjectives.csv',
                './prompt_generation/data/raw/OEWS21_OccupationsDetailedView.csv']
    },
    packages=setuptools.find_packages(),
    entry_points ={
            'console_scripts': [
                'facia=saac.main:main'
            ]
        },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.8',
    install_requires=required
)
