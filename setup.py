import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="saac",
    version="0.0.1",
    author="TRSS",
    author_email="valeria.rozenbaum@trssllc.com",
    description="Framework for text-to-image auditing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="//https://github.com/TRSS-Research/SAAC",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.8',
    install_requires=required
)
