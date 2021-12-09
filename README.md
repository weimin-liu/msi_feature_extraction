# MSI Feature Extraction

Clean mass spectrometry imaging dataset and extract geologically meaningful features

## Prerequisite
Before using the workflow, proprietary mass spectrometry data format (e.g., .D from Bruker) needs to be exported as plain text file (represented as `da_exported_txt` in the following examples). Only the coordinates and the centroid mass-to-charge ratios along with the peak intensity are needed from each spectrum.

Python >= 3.5 is needed, and the required library is listed in requirements.txt.

The package has been tested on both Windows (Windows 10), OSX, and Linux (Archlinux).

## Instruction
### Mass calibration
Dataset should be calibrated first if it hasn't been calibrated yet. Currently, only a slightly modified version of single-point lock mass calibration is available in this package.

````python
from src.mfe import suggest_calibrates, SimpleFallbackCalibrate
from src.mfe import msi_from_txt

# get a list of the most abundant peaks in the dataset

candidates, _ = suggest_calibrates(da_exported_txt)

# create a dictionary to store the dataset
msi = msi_from_txt(da_exported_txt)

sfc = SimpleFallbackCalibrate()

# feed the list of calibrates to SimpleFallbackCalibrate. Assign each spectrum with a calibrate. The calibrate is decided as follows: first try to use the first calibrate in the list in all spectra, if the calibrate is missing in some spectra, it will then try to calibrate those spectra with the second calibrate in the list, and so on, until the spectra are all calibrated or the calibrate list is exhausted.
sfc.fit(msi, candidates)

# do the actual calibration on the dataset
msi_calibrated = sfc.transform(msi)
````

### Align peaks into discrete mass bins

Currently, the discrete mass bins are evenly spaced with user designated interval.

````python
from src.mfe import create_feature_table

feature_table = create_feature_table(msi_calibrated)
````

A 2D table will be produced in this step, with columns being the name of mass bins (m/z ratios), and each row representing one spot.

### Pick peaks using grey-level co-occurrences matrix
No peak has been dropped until this step, grey-level co-occurrences matrix (GLCM) are used to detect how structured are those ion images and rank them.

````python
from src.mfe import get_peak_ranks

t_df, deflated_arr = get_peak_ranks(feature_table)
````
The result contains the ranked peaks with its corresponding ion image, manual examination is needed to decide a threshold (`th`) above which the peaks are preserved.

````python
from src.mfe import sel_peak_by_rank

feature_table, ims = sel_peak_by_rank(t_df, deflated_arr, feature_table, th)
````
### Feature extraction using non-negative matrix factorization

````python
from src.mfe import rank_estimate, nmf

# first detect the appropriate rank for the data, the list of images are used here instead of the feature table, because the images have already been normalized with quantiles removed.
rank_candidates = list(range(2, 20))

rank_estimate(rank_candidates, ims)

# then do the factorization with an appropriate rank `rk`, getting the basis matrix and the coeffcients
basis, components = nmf(ims, feature_table, rk)

# to get the co-localization molecular network, n_run >1 must be set
basis, components, G = nmf(ims, feature_table, rk, n_run=20)
````

## Notes:

- Setting a higher `beta` parameter in Nimfa.Snmf() results in a more sparse matrix, useful in identify key peaks in reconstructed mass spectra.

## Credits

- [francisbrochu/msvlm](https://github.com/francisbrochu/msvlm) for the `Spectrum` Class

- [mims-harvard/nimfa](https://github.com/mims-harvard/nimfa) for the NMF analysis
