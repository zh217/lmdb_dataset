import os
import lmdb
import pyarrow

from torch.utils.data import Dataset

from .utils import encode_key


class LMDBDataset(Dataset):
    def __init__(self, db_path, subset_n=None, map_size=10):
        super().__init__()
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False,
                             map_size=map_size)
        with self.env.begin(write=False) as txn:
            self.length = pyarrow.deserialize(txn.get(b'__len__'))
            try:
                self.keys = pyarrow.deserialize(txn.get(b'__keys__'))
            except:
                self.keys = None  # [txt_utils.encode_key(i) for i in range(self.length)]
        if subset_n:
            self.length = subset_n

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
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
