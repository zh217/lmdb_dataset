import os
import lmdb
import pyarrow
import random

from torch.utils.data import Dataset

from .utils import encode_key


class LMDBDataset(Dataset):
    def __init__(self, db_path, subset_n=None, readahead=True, restart_every=-1):
        super().__init__()
        self.db_path = db_path
        self.use_count = 0
        assert restart_every != 0
        self.restart_every = restart_every
        self.readahead = readahead
        self.db = None

        with self.get_db().begin(write=False) as txn:
            self.length = pyarrow.deserialize(txn.get(b'__len__'))
            try:
                self.keys = pyarrow.deserialize(txn.get(b'__keys__'))
            except:
                self.keys = None  # [txt_utils.encode_key(i) for i in range(self.length)]
        if subset_n:
            self.length = subset_n

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
            if self.keys:
                k = self.keys[index]
            else:
                k = encode_key(index)
            byteflow = txn.get(k)
        return pyarrow.deserialize(byteflow)

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

    def shuffle(self):
        indices = list(range(len(self)))
        random.shuffle(indices)
        for idx in indices:
            yield self[idx]
