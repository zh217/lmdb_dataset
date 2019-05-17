import os
import pyarrow
import lmdb

from lmdb_dataset.utils import encode_key
from lmdb_dataset.dataset import LMDBDataset


class LMDBDatasetWriter:
    def __init__(self, dataset_path, map_size=1099511627776 * 10):
        try:
            os.makedirs(dataset_path)
            self.count = 0
        except FileExistsError:
            old_dset = LMDBDataset(dataset_path)
            self.count = len(old_dset)
            del old_dset

        self.db = lmdb.open(dataset_path, subdir=True,
                            map_size=map_size, readonly=False,
                            meminit=False, map_async=True, lock=False)

    def get_next_key(self):
        next_key = encode_key(self.count)
        self.count += 1
        return next_key

    def write_len(self, txn):
        txn.put(b'__len__', pyarrow.serialize(self.count).to_buffer())

    def write_data(self, elements, *, commit_every=100):
        it = iter(elements)

        processing = True

        while processing:
            with self.db.begin(write=True) as txn:
                for _ in range(commit_every):
                    try:
                        payload = pyarrow.serialize(next(it)).to_buffer()
                        txn.put(self.get_next_key(), payload)
                    except StopIteration:
                        processing = False
                self.write_len(txn)

    def close(self):
        self.db.sync()
        self.db.close()
