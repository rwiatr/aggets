import multiprocessing as mp
from multiprocessing import shared_memory


def execute(df, fn, threads=None):
    shms = [shared_memory.SharedMemory(size=array.size()) for array in nps]
    threads = mp.cpu_count() - 1 if threads is None else threads

    with mp.Pool(processes=threads) as pool:
        pool.apply()