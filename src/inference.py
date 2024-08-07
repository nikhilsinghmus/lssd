

import torch
from tqdm import tqdm
from utils import distributed, config
from utils.utils import load_checkpoint, Buffer, set_random_seed
from datasets import build_dataset
from models import build_model



@torch.no_grad()
def _feature_extraction(cfg, data_loader, model):
    model.eval()
    helper = Buffer(data_loader.dataset)
    for data in tqdm(data_loader, desc=f'Feature Extraction:{data_loader.dataset.mtype}', dynamic_ncols=True):
        with torch.cuda.amp.autocast(enabled=True):
            # forward pass through backbone
            pred = model(data)
            # aggregate input metadata and output
            helper.collect(data['idx'], pred)
    # save output
    helper.save_feature(cfg.OUTPUT_DIR)



def feature_extraction(cfg):
    # Setting up distributed training
    distributed.init(cfg)

    # Partially ensuring reproducibility
    set_random_seed(cfg)

    # Creating dataset for feature extraction
    dataset = build_dataset(cfg)

    # Creating model instance
    model = build_model(cfg)

    # Load model weights
    load_checkpoint(model, cfg.TEST.CHECKPOINT_FILE_PATH)

    # Run inference one modality at a time
    for data_loader in dataset.__dict__.values():
        _feature_extraction(cfg, data_loader, model)


def main():
    args = config.parse_args()
    cfg = config.load_config(args)
    distributed.launch(
        cfg=cfg,
        init_method=args.init_method,
        func=feature_extraction,
        )

if __name__ == "__main__":
    main()
