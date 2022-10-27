import unittest

from src.mfe.from_txt import msi_from_txt, get_ref_peaks, create_feature_table


class TestBasic(unittest.TestCase):
    def test_msi_from_txt(self):
        msi = msi_from_txt('test_data/da_export.txt')
        ref_peaks = get_ref_peaks(msi, peak_th=0.2)
        feature_table, err_table = create_feature_table(msi, ref_peaks[0.2], normalization='median')
        assert len(msi) == 9
        assert len(ref_peaks[0.2]) != 0
        print(feature_table)
        assert len(feature_table) == len(msi)
        assert len(feature_table.columns) == len(ref_peaks[0.2]) + 2


if __name__ == '__main__':
    unittest.main()
