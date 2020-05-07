import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='images_to_dataset',
    version='0.3',
    packages=setuptools.find_packages(),
    url='https://github.com/GianmarcoMidena/images-to-dataset',
    license='MIT License',
    author='Gianmarco Midena',
    author_email='gianmarco.midena@gmail.com',
    description='A tool for building a dataset from a set of images',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
    ]
)
