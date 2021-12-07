# msi_feature_extraction Clean and extract feature from MALDI Imaging Data   ## Pre-Step(s) ...  Export mass spectrometry data from proprietary mass spectrometry data format (e.g., .D folder for Bruker). The resulting file would be a plain text file with coordinates and centroid mass-to-charge ratios along with the peak intensity.  ## Steps 
1. Mass calibration

Currently, only single-point lock mass calibration is available.

"""
# get the most abundant peaks in the whole dataset first

from mfe.calibration import suggest_calibrates, SimpleFallbackCalibrate
from mfe.from_txt import msi_from_txt 

candidates, _ = suggest_calibrates(da_exported_txt)

# supply the list of calibrates to SimpleFallbackCalibrate. It will first calibrate all spectra with the first calibrate in the list, if the calibrate is missing in some spectra, it will then calibrate those spectra with the second calibrate in the list, and so on, until the spectra are all calibrated or the calibrate list is exhausted.

msi = msi_from_txt(da_exported_txt)

sfc = SimpleFallbackCalibrate()

sfc.fit(msi, candidates)

msi_calibrated = sfc.transform(msi)

"""


2. Align peaks into discrete mass bins

Currently, the discrete mass bins are evenly spaced with user designated interval.
"""
from mfe.from_txt import get_feature_table
 
feature_table = get_feature_table(da_exported_txt)
"""

3. Pick peaks using grey-level co-occurences matrix


"""
from mfe.peak_picking import get_peak_ranks

t_df, deflated_arr = get_peak_ranks(feature_table)

"""
The result contains the ranked peaks with its corresponding ion image, manual examination is needed to decide a threshold above which the peaks are preserved.
  
4. Feature extraction using non-negative matrix factorization

"""
from mfe.feature import rank_estimate

# first detect the appropriate rank for the data
rank_estimate(im)

# then get the basis and coefficients

"""
  ## Credits  [francisbrochu/msvlm](https://github.com/francisbrochu/msvlm) for the well-designed `Spectrum` Class 
