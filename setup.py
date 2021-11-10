from setuptools import setup

setup(
    name='mfe',
    version='0.0.1',
    install_requires=['numpy', 'pandas', 'matplotlib', 'scipy', 'scikit-learn'],
    packages=['mfe.src.util', 'mfe.src.vis'],
    url='https://github.com/weimin-liu/msi_feature_extraction',
    license='MIT License',
    author='Weimin Liu',
    author_email='wliu@marum.de',
    description='This module allows to extract binned features from MALDI imaging spectrometry data'
)
