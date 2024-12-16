import torch
import numpy as np

if __name__ == "__main__":

    size = 1024 * 1024 * 8
    tensor = torch.randn(size, pin_memory=True, dtype=torch.float32, device="cpu")
    gpu_tensor = torch.randn(size, dtype=torch.float32, device="cuda:0")

    # tensor.copy_(gpu_tensor)

    # warm up
    for _ in range(5):
        gpu_tensor.copy_(tensor)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    gpu_tensor.copy_(tensor)
    end_event.record()
    torch.cuda.synchronize()
    time_ms = start_event.elapsed_time(end_event)
    print(f"time:{time_ms} ms")
    print(f"bandwidth:{(size * 4) / (time_ms * 1e-3) / 1e9} GB/s")
    print(gpu_tensor)
