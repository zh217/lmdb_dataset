from multiprocessing.pool import Pool
from threading import Semaphore


def encode_key(c):
    return ('%012d' % c).encode('ascii')


def parallel_map(iterable, func, n_procs, *, chunksize=1, buffer_ratio=2, **pool_args):
    semaphore = Semaphore(int(chunksize * n_procs * buffer_ratio))

    def pre_throttle(it):
        for x in it:
            semaphore.acquire()
            yield x

    with Pool(processes=n_procs, **pool_args) as pool:
        for result in pool.imap_unordered(func, pre_throttle(iterable), chunksize=chunksize):
            semaphore.release()
            yield result


def iter_dataset(dset):
    """
    Necessary because native pytorch implementation of __iter__ on Dataset is problematic
    :param dset:
    :return:
    """
    for i in range(len(dset)):
        yield dset[i]
