import os
import pyarrow
import lmdb

from lmdb_dataset.utils import encode_key
from lmdb_dataset.dataset import LMDBDataset


class LMDBDatasetWriter:
    def __init__(self, dataset_path, map_size=1099511627776 * 10, restart_every=100):
        try:
            os.makedirs(dataset_path)
            self.count = 0
        except FileExistsError:
            old_dset = LMDBDataset(dataset_path)
            self.count = len(old_dset)
            del old_dset

        self.dataset_path = dataset_path
        self.map_size = map_size
        self.use_count = 0
        self.db = None
        assert restart_every != 0
        self.restart_every = restart_every

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
            with self.get_db().begin(write=True) as txn:
                for _ in range(commit_every):
                    try:
                        payload = pyarrow.serialize(next(it)).to_buffer()
                        txn.put(self.get_next_key(), payload)
                        del payload
                    except StopIteration:
                        processing = False
                self.write_len(txn)
        self.close()

    def get_db(self):
        if self.use_count == self.restart_every:
            self.close()
            return self.get_db()

        if not self.db:
            self.db = lmdb.open(self.dataset_path, subdir=True,
                                map_size=self.map_size, readonly=False,
                                meminit=False, map_async=True, lock=True)
        self.use_count += 1
        return self.db

    def close(self):
        if self.db:
            self.db.sync()
            self.db.close()
        self.db = None
        self.use_count = 0
