# mfe

Yet another automated mass spectrometry imaging data processing package, but for **geological samples**. 

## Prerequisite

Before using the workflow, proprietary mass spectrometry data format (e.g., .D from Bruker) needs to be exported as plain text file (represented as `da_exported_txt` in the following examples). Only the coordinates and the centroid mass-to-charge ratios along with the peak intensity are needed from each spectrum.

Python >= 3.5 is needed, and the required library is listed in requirements.txt.

This package is OS-independent

## Installation

Just run the following command and the package with all dependecies will be installed.

````bash
pip install git+https://github.com/weimin-liu/msi_feature_extraction.git
````

## Instruction

### Mass calibration

Dataset should be calibrated first if it hasn't been calibrated yet. Currently, a quadratic mass error calibration function (`mfe.quadratic_calibration`) is available in this package. Just provide the list of calibrants, and a quadratic function will be fitted to calibrate the spectrum. 

### Bin-wise KDE for reference peak detection and peak alignment

Reference peaks that are used for spectra alignment are detected in the evenly spaced discrete mass bins, with a peak picking threshold value `pth`. 

The following is an example of how to create a feature table from the MSI spectra with ion intensities normalized by the median peak intensity of each individual spectrum.

````python
from mfe.from_txt import msi_from_txt, get_ref_peaks, create_feature_table

# create a msi object from the exported txt file
spectra = msi_from_txt(da_exported_txt)

# get the reference peaks, with a peak picking threshold of pth, and pth is a list
ref = get_ref_peaks(spectra, peak_th=pth)

# create a feature table with the reference peaks
feature_table = create_feature_table(spectra, ref, normalization='median')
````

A 2D table will be produced in this step, with columns being the name of the reference peaks (*m/z* ratios), and each row representing one laser spot.

### Pick peaks using grey-level co-occurrences matrix (optional)
No peak has been dropped until this step, grey-level co-occurrences matrix (GLCM) are used to detect how structured are those ion images and rank them.

The following is an example of comparing the GLCM features in MSI dataset with those in the X-radiophotograph.

````python
from mfe.peak_picking import GLCMPeakRanking

glcm = GLCMPeakRanking(q_val=8)
glcm.fit(feature_table, [1, 2, 3, 4, 5], [np.pi / 6, 0, -np.pi / 6, np.pi / 2, -np.pi / 2, np.pi / 4, -np.pi / 4])

results = glcm.results
results = pd.DataFrame(results)

````

### Feature extraction using non-negative matrix factorization

````python
from mfe.feature import repeated_nmf

# first detect the appropriate rank for the data, the list of images are used here instead of the feature table, because the images have already been normalized with quantiles removed.

rank_candidates = list(range(2, 20))
summary = {}
for rank in rank_candidates
    summary[rank] = repeated_nmf(ims, rank, 30, init='random', max_iter=3000)
````

## TODO:

- [ ] Add a more sophisticated piecewise function for mass calibration
- [ ] Add an image registration function (Affine transformation already existed in the code)


## Credits

- [francisbrochu/msvlm](https://github.com/francisbrochu/msvlm) for the `Spectrum` Class

