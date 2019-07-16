from torch.utils.data import Dataset


class InterleaveDataset(Dataset):
    def __init__(self, *datasets):
        super().__init__()
        self.datasets = list(datasets)
        self.lengths = [len(d) for d in self.datasets]
        self.length = sum(self.lengths)
        self.indices = []
        for i, l in enumerate(self.lengths):
            self.indices += list((i, k) for k in range(l))
        self.indices.sort(key=lambda mi: (mi[1] / self.lengths[mi[0]], mi[0]))

    def __getitem__(self, index):
        db_idx, item_idx = self.indices[index]
        return self.datasets[db_idx][item_idx]

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + repr(self.datasets)


class ConcatDatabase(Dataset):
    def __init__(self, *datasets):
        super().__init__()
        self.datasets = list(datasets)
        self.lengths = [len(d) for d in self.datasets]
        self.length = sum(self.lengths)
        self.indices = []
        for i, l in enumerate(self.lengths):
            self.indices += list((i, k) for k in range(l))
        self.last_db = None

    def __getitem__(self, index):
        db_idx, item_idx = self.indices[index]
        cur_db = self.datasets[db_idx]
        if self.last_db and self.last_db != cur_db:
            self.last_db.close()
        self.last_db = cur_db
        return cur_db[item_idx]

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + repr(self.datasets)


if __name__ == '__main__':
    ds = InterleaveDataset(
        list(range(10)),
        list('abc'),
        list('UVWXYZ')
    )
    print(len(ds))
    for i in range(len(ds)):
        print(i, ds[i])
