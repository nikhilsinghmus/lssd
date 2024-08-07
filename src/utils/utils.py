import os
import torch
import numpy as np
import random
import json
from fvcore.common.file_io import PathManager as pm
from fvcore.nn.distributed import differentiable_all_gather
from .distributed import is_master_proc


# helper function to load the model checkpoints
def load_checkpoint(model, path_to_checkpoint):
    assert pm.exists(path_to_checkpoint), "Checkpoint '{}' not found".format(path_to_checkpoint)
    with pm.open(path_to_checkpoint, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")

    # only keep the backbone weights
    checkpoint["model"] = {k.replace('backbone.','') : v \
                           for k,v in checkpoint["model"].items() if k.startswith('backbone.')}
    # load the weights
    model.module.load_state_dict(checkpoint["model"], strict=True)


# helper function to partially make inference reproducible
def set_random_seed(cfg):    
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    random.seed(cfg.RNG_SEED)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False


# helper function to keep and store the embeddings
class Buffer(object):
    def __init__(self, dataset):

        self.dataset = dataset

    def collect(self, idx, y):
        if not hasattr(self, 'buffer'):
            self.buffer = {k : torch.zeros(
                (len(self.dataset), *v.shape[1:]), dtype=v.dtype) \
                for k,v in y.items()}
        # save after gathering from all devices
        idx = torch.cat(differentiable_all_gather(idx.cuda()))
        for k in y.keys():
            self.buffer[k][idx] = torch.cat(
                differentiable_all_gather(y[k].contiguous())
                ).cpu()

    def save_feature(self, output_dir):
        if is_master_proc():
            # create output directory
            out_dir = os.path.join(output_dir, self.dataset.mtype)
            if not pm.exists(out_dir):
                pm.mkdirs(out_dir)
            # save metadata
            with pm.open(out_dir + "/metadata.json", 'w') as f:
                json.dump(self.dataset.items_dict, f, indent=4)
            # save features
            for k, v in self.buffer.items():
                with pm.open(out_dir + f"/{k}.pt", "wb") as f:
                    torch.save(v, f)



# helper function to load individual embedding tensors for each input file
def postprocess(output_dir, mtype):
    # this is the naming pattern during inference
    dir = f"{output_dir}/{mtype}"
    assert os.path.exists(dir)
    # getting metadata and actual features
    metadata = json.load(open(f"{dir}/metadata.json"))
    tensor = torch.load(f"{dir}/{mtype}.pt")
    # each instance may compose of more than one clip at inference
    split_size_or_sections = list(map(len, metadata['range'])) # number of clips per asset
    # here we have a list of feature parts in MxD(audio)/Mx5xD(video) where M varies depending on the shot length
    feature_parts = torch.split(tensor, split_size_or_sections,dim=0)
    # regroup
    assert len(feature_parts) == len(metadata['filepath'])
    return list(zip(metadata['filepath'], feature_parts))
