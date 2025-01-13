import torch
import torch.distributed as dist


class Parallel:
    rank: int
    world_size: int
    device: torch.device

    def __init__(self):
        dist.init_process_group(backend="nccl")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")

    def destroy(self):
        dist.destroy_process_group()

    def is_first_rank(self):
        return self.rank == 0

    def reduce(
        self,
        input_: torch.Tensor,
        dst: int = 0,
        op=dist.ReduceOp.SUM,
    ):
        if self.world_size == 1:
            return
        dist.reduce(input_, dst=dst, op=op)

    def gather(
        self,
        input_: torch.Tensor,
        dst: int = 0,
        dim: int = -1,
    ) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        if self.rank == dst:
            gather_list = [torch.empty_like(input_) for _ in range(self.world_size)]
        else:
            gather_list = None
        dist.gather(input_, gather_list=gather_list, dst=dst)
        if self.rank == dst:
            output = torch.cat(gather_list, dim=dim)
        else:
            output = None
        return output
