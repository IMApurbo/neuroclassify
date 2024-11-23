from setuptools import setup, find_packages

setup(
    name='neuroclassify',
    version='3.0',
    description='A simple image classification package using deep learning.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='AKM Korishee Apurbo',
    author_email='bandinvisible8@gmail.com',
    url='https://github.com/IMApurbo/neuroclassify',  # Replace with your GitHub URL or project page
    packages=find_packages(),
    install_requires=[
        'tensorflow>=2.0',  # TensorFlow is required
        'numpy>=1.18.5',    # Numpy is required for TensorFlow
        'matplotlib>=3.0',   # Matplotlib is required for image display
        'Pillow>=7.0.0',     # Pillow is required for image processing
        'scikit-learn>=0.23.2',  # scikit-learn for utility functions like splitting data
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # The package requires Python 3.6 or higher
    entry_points={
        'console_scripts': [
            # Command line tools can be added here if needed in future
        ],
    },
    include_package_data=True,  # Include additional files, like the README
    zip_safe=False,
)
