from multiprocessing import Pool
import tempfile
import unittest

import numpy as np


class TestPool(unittest.TestCase):
    def setUp(self):
        pass


def get_chunks(data):
    for row in data:
        yield row

def setUp(n_samples=1000, n_features=100):
    file_pointer = tempfile.TemporaryFile()

    arr = np.random.randn(n_samples, n_features)

    np.save(file_pointer, arr)

    # Simulate closing and re-opening the file
    file_pointer.seek(0)

    return file_pointer


def do_work(chunk):
    return chunk.mean()


def main():
    n_samples, n_features = 1000, 300
    file_pointer = setUp()
    data = np.load(file_pointer)
    pool = Pool()
    results = pool.imap(do_work, get_chunks(data))
    results = np.fromiter(results, dtype=np.float)

    print(results)
    print(results.shape)


if __name__ == "__main__":
    main()
    # unittest.main()
