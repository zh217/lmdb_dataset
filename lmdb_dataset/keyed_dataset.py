import os
import lmdb
import pyarrow
import random

from .utils import encode_key


class LMDBKeyedDataset:
    def __init__(self, db_path, readahead=True, restart_every=-1, transform=lambda item: item):
        super().__init__()
        self.db_path = db_path
        self.use_count = 0
        assert restart_every != 0
        self.restart_every = restart_every
        self.readahead = readahead
        self.db = None
        self.transform = transform

    def get_db(self):
        if self.use_count == self.restart_every:
            self.close()
            return self.get_db()

        if not self.db:
            self.db = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path),
                                readonly=True, lock=False,
                                readahead=self.readahead, meminit=False)
        self.use_count += 1
        return self.db

    def close(self):
        if self.db:
            self.db.close()
        self.db = None
        self.use_count = 0

    def __getitem__(self, index):
        with self.get_db().begin(write=False) as txn:
            byteflow = txn.get(index.encode('utf-8'))
        item = pyarrow.deserialize(byteflow)
        return self.transform(item)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'
