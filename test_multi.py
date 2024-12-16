import torch.multiprocessing as mp
import torch
import time


def worker(q: mp.Queue, i):
    tensor = q.get()
    tensor[i] = i
    time.sleep(1)


def main():
    # Set multiprocessing start method
    mp.set_start_method("forkserver")

    # Create a large tensor of size ~1GB
    size = 1024 * 1024 * 1024  # Approximately 1GB for float32
    tensor = torch.empty(size, dtype=torch.float32).cuda(1)
    time.sleep(10)


    # Create processes
    num_processes = 10
    processes = []
    q = mp.Queue()
    for i in range(num_processes):
        p = mp.Process(target=worker, args=(q, i))
        processes.append(p)


    for p in processes:
        p.start()

    for i in range(num_processes):
        q.put(tensor)

    for p in processes:
        p.join()

    # Check the last 10 positions of the tensor
    print(tensor[:10])


if __name__ == "__main__":
    main()
