from setuptools import setup, find_packages

setup(
    name='neuroclassify',
    version='1.0',
    description='A simple image classification package using deep learning.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='AKM Korishee Apurbo',
    author_email='bandinvisible8@gmail.com',
    url='https://github.com/IMApurbo/neuroclassify',  # Update with your GitHub URL
    packages=find_packages(),  # Automatically find the packages
    install_requires=[
        'tensorflow>=2.0',
        'numpy',
        'matplotlib',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
