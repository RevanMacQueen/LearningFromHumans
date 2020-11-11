import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LFH", # Replace with your own username
    version="0.0.0",
    author="Revan MacQueen",
    author_email="revan@ualberta.ca",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RevanMacQueen/LearningFromHumans",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)