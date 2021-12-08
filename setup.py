from setuptools import setup

setup(
    name='mfe',
    version='0.0.1',
    install_requires=['numpy', 'pandas', 'matplotlib', 'scipy', 'scikit-learn','scikit-image'],
    packages=['mfe.src'],
    url='https://github.com/weimin-liu/msi_feature_extraction',
    license='MIT License',
    author='Weimin Liu',
    author_email='wliu@marum.de',
    description='Clean mass spectrometry imaging dataset and extract geologically meaningful features'
)
