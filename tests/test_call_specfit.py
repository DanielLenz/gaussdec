import unittest
from pathlib import Path
import string
import random
from tempfile import TemporaryFile

import numpy as np
import healpy as hp

from myhelpers import misc

import site

site.addsitedir(misc.bpjoin("gaussdec"))

from src.decompose import call_specfit as cs


def random_string(length: int = 10) -> str:
    _str = "".join(
        random.choice(string.ascii_letters + string.digits) for _ in range(length)
    )

    return _str


class TestCreateTables(unittest.TestCase):
    def test_baseline(self):
        with TemporaryFile() as temp_file:
        filepath = Path("/tmp/").joinpath(random_string(10))

        cs.create_tables(filepath, clobber=True)


class TestGetRowIndex(unittest.TestCase):
    def test_hpxindices(self):
        """
        Testcase if an .npy file with indices is given
        """

        input_indices = np.random.randint(0, 1000, size=100)
        with TemporaryFile() as temp_file:
            np.save(temp_file, input_indices)
            temp_file.seek(0)

            iterator = cs.get_row_index(
                n_samples=100, hpxindices=temp_file, n_spectra_in_survey=100
            )

            np.testing.assert_array_equal(
                np.fromiter(iterator, dtype=int), input_indices
            )

    def test_random_samples(self):
        """
        Testcase for a small number of random samples
        """
        nside = 64
        n_spectra_in_survey = hp.nside2npix(nside)

        n_samples = 50

        iterator = cs.get_row_index(
            n_samples=n_samples, hpxindices=None, n_spectra_in_survey=n_spectra_in_survey
        )
        indices = np.fromiter(iterator, dtype=int)

        # Check that we get the right number of samples
        self.assertEqual(n_samples, indices.size)

        # Check that all indices are within the expected range
        self.assertGreaterEqual(indices.min(), 0)
        self.assertLess(indices.max(), n_spectra_in_survey)

        # Check that we have a reasonable number of unique indices
        n_unique_indices = np.unique(indices).size
        self.assertGreaterEqual(n_unique_indices, n_samples // 2)



if __name__ == "__main__":
    unittest.main()
