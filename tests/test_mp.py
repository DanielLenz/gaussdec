import numpy as np
from multiprocessing import Pool, Process, cpu_count, Queue
from time import sleep

def target(nums, queue):

    for num in nums:
        queue.put({num: np.sqrt(num)})


def main():

    n_cpus = cpu_count()
    total_load = np.arange(100)
    loads = np.array_split(total_load, n_cpus)
    queue = Queue()

    jobs = []
    for i, load in enumerate(loads):
        p = Process(target=target, args=(load, queue))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    while not queue.empty():
        item = queue.get(block=False)
        print(item)


if __name__ == "__main__":
    main()
