
import torch
import torch.distributed as dist

_LOCAL_PROCESS_GROUP = None


def is_master_proc():
    if dist.is_initialized():
        return dist.get_rank() % dist.get_world_size() == 0
    else:
        return True

def init(cfg):
    if cfg.DIST.NUM_GPUS * cfg.DIST.NUM_SHARDS == 1:
        return
    for i in range(cfg.DIST.NUM_SHARDS):
        ranks_on_i = list(range(i * cfg.DIST.NUM_GPUS, (i + 1) * cfg.DIST.NUM_GPUS))
        pg = dist.new_group(ranks_on_i)
        if i == cfg.DIST.SHARD_ID:
            global _LOCAL_PROCESS_GROUP
            _LOCAL_PROCESS_GROUP = pg

def run(local_rank, num_proc, func, init_method, shard_id, num_shards, backend, cfg):
    world_size = num_proc * num_shards
    rank = shard_id * num_proc + local_rank
    try:
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            )
    except Exception as e:
        raise e
    torch.cuda.set_device(local_rank)
    func(cfg)


def launch(cfg, init_method, func, daemon=False):
    torch.multiprocessing.spawn(
        run,
        nprocs=cfg.DIST.NUM_GPUS,
        args=(
            cfg.DIST.NUM_GPUS,
            func,
            init_method,
            cfg.DIST.SHARD_ID,
            cfg.DIST.NUM_SHARDS,
            cfg.DIST.BACKEND,
            cfg,
            ),
        daemon=daemon,
    )