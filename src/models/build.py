import torch
from fvcore.common.registry import Registry

MODEL_REGISTRY = Registry("MODEL")

def build_model(cfg):
    # we do not need the projection head for inference
    model = build_backbone(cfg)
    
    # push to gpu
    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)
    model = torch.nn.parallel.DistributedDataParallel(
            module=model, 
            device_ids=[cur_device], 
            output_device=cur_device,
            )

    return model


def build_backbone(cfg):
    model = {}
    if cfg.DATA.USE_VIDEO:
        model['video'] = MODEL_REGISTRY.get(cfg.MODEL.VIDEO)(cfg, mtype='video')
    if cfg.DATA.USE_AUDIO:
        model['audio'] = MODEL_REGISTRY.get(cfg.MODEL.AUDIO)(cfg, mtype='audio')
            
    model = {k: torch.nn.SyncBatchNorm.convert_sync_batchnorm(v) for k, v in model.items()}

    return Backbone(model)


class Backbone(torch.nn.Module):
    def __init__(self, model_dict):
        super(Backbone, self).__init__()
        self.net = torch.nn.ModuleDict(model_dict)

    def forward(self, input):
        # find the right tower of the network, given the modality of the input
        output = {}
        for mtype in self.net.keys():
            if mtype in input:
                output[mtype] = self.net[mtype](input[mtype], inference=True)
        return output
