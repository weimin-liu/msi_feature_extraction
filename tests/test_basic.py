import unittest

from src.mfe.from_txt import msi_from_txt, get_ref_peaks, create_feature_table
from src.mfe.peak_picking import GLCMPeakRanking, de_flatten
import numpy as np
from src.mfe.feature import repeated_nmf


class TestBasic(unittest.TestCase):
    def test_msi_from_txt(self):
        msi = msi_from_txt('test_data/da_export.txt')
        ref_peaks = get_ref_peaks(msi, peak_th=0.2)
        feature_table, err_table = create_feature_table(msi, ref_peaks[0.2], normalization='median')
        assert len(msi) == 9
        assert len(ref_peaks[0.2]) != 0
        assert len(feature_table) == len(msi)
        assert len(feature_table.columns) == len(ref_peaks[0.2]) + 2

        glcm = GLCMPeakRanking(q_val=8)
        glcm.fit(feature_table, [1, 2, 3, 4, 5],
                 [np.pi / 6, 0, -np.pi / 6, np.pi / 2, -np.pi / 2, np.pi / 4, -np.pi / 4])
        results = glcm.results
        assert len(results) == len(ref_peaks[0.2])

        ims = list()

        for mz in feature_table.drop(['x', 'y'], axis=1).columns:
            im = de_flatten(feature_table[['x', 'y']].to_numpy(), feature_table[mz].to_numpy().flatten())
            ims.append(im)

        ims = [im.flatten() for im in ims]

        ims = np.array(ims)

        ims = ims.T

        rank_candidates = list(range(2, 5))
        summary = {}
        for rank in rank_candidates:
            summary[rank] = repeated_nmf(ims, rank, 3, init='random', max_iter=30)
        assert len(summary) == len(rank_candidates)


if __name__ == '__main__':
    unittest.main()
