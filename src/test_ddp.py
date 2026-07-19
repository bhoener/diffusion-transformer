import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

def ddp_setup() -> None:
    ddp_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(ddp_rank)
    
    dist.init_process_group(backend="nccl")

def ddp_cleanup() -> None:
    dist.destroy_process_group()

def main():
    print("hello world")
    ddp_setup()
    print("ddp setup")
    ddp_rank = int(os.environ["LOCAL_RANK"])
    print("ddp local rank:", ddp_rank)
    ddp_world_size = int(dist.get_world_size())
    print("ddp world size:", ddp_world_size)
    
    ddp_global_rank = dist.get_rank()
    print("ddp global rank:", ddp_global_rank)

    print(f"Hello from GPU {ddp_rank}")
    
    a = torch.randn(2, 2).to(ddp_rank)
    
    print(a)
    
    ddp_cleanup()
    print(f"Finished running on rank {ddp_rank}")

if __name__ == "__main__":
    main()