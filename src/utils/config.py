import argparse
from fvcore.common.config import CfgNode


# Config definition
_C = CfgNode()

#------------------------------
# Testing options.
#------------------------------
_C.TEST = CfgNode()
# Dataset.
_C.TEST.DATASET = ""
# Batch size per-GPU. Adjust wrt your max GPU memory
_C.TEST.BATCH_SIZE = 8
# Path to the checkpoint to load the initial weight.
_C.TEST.CHECKPOINT_FILE_PATH = ""
# Duration (in seconds) of clips used in inference
_C.TEST.CLIP_LEN = 3.0

#------------------------------
# Distributed Compute options
#------------------------------
_C.DIST = CfgNode()
# Number of GPUs.
_C.DIST.NUM_GPUS = 2
# Number of machines to use for the job.
_C.DIST.NUM_SHARDS = 1
# The index of the current machine.
_C.DIST.SHARD_ID = 0
# Distributed backend.
_C.DIST.BACKEND = "nccl"
# Number of data loader workers per training process.
_C.DIST.NUM_WORKERS = 8


#------------------------------
# Model options
#------------------------------
_C.MODEL = CfgNode()
# Backbone architecture of video stream
_C.MODEL.VIDEO = "MViT"
# Backbone architecture of audio stream
_C.MODEL.AUDIO = "MViT"


#------------------------------
# MViT options
#------------------------------
_C.MVIT = CfgNode()
# Options include `conv`, `max`.
_C.MVIT.MODE = "conv"
# If True, perform pool before projection in attention.
_C.MVIT.POOL_FIRST = False
# If True, use cls embed in the network, otherwise don't use cls_embed in transformer.
_C.MVIT.CLS_EMBED_ON = True
# Kernel size for patchtification.
_C.MVIT.PATCH_KERNEL = [3, 7, 7]
# Stride size for patchtification.
_C.MVIT.PATCH_STRIDE = [2, 4, 4]
# Padding size for patchtification.
_C.MVIT.PATCH_PADDING = [2, 4, 4]
# If True, use 2d patch, otherwise use 3d patch.
_C.MVIT.PATCH_2D = False
# Base embedding dimension for the transformer.
_C.MVIT.EMBED_DIM = 96
# Base num of heads for the transformer.
_C.MVIT.NUM_HEADS = 1
# Dimension reduction ratio for the MLP layers.
_C.MVIT.MLP_RATIO = 4.0
# If use, use bias term in attention fc layers.
_C.MVIT.QKV_BIAS = True
# Drop path rate for the tranfomer.
_C.MVIT.DROPPATH_RATE = 0.1
# The initial value of layer scale gamma. Set 0.0 to disable layer scale.
_C.MVIT.LAYER_SCALE_INIT_VALUE = 0.0
# Depth of the transformer.
_C.MVIT.DEPTH = 16
# Normalization layer for the transformer. Only layernorm is supported now.
_C.MVIT.NORM = "layernorm"
# Dimension multiplication at layer i. If 2.0 is used, then the next block will increase
# the dimension by 2 times. Format: [depth_i: mul_dim_ratio]
_C.MVIT.DIM_MUL = []
# Head number multiplication at layer i. If 2.0 is used, then the next block will
# increase the number of heads by 2 times. Format: [depth_i: head_mul_ratio]
_C.MVIT.HEAD_MUL = []
# Stride size for the Pool KV at layer i.
# Format: [[i, stride_t_i, stride_h_i, stride_w_i], ...,]
_C.MVIT.POOL_KV_STRIDE = []
# Initial stride size for KV at layer 1. The stride size will be further reduced with
# the raio of MVIT.DIM_MUL. If will overwrite MVIT.POOL_KV_STRIDE if not None.
_C.MVIT.POOL_KV_STRIDE_ADAPTIVE = None
# Stride size for the Pool Q at layer i.
# Format: [[i, stride_t_i, stride_h_i, stride_w_i], ...,]
_C.MVIT.POOL_Q_STRIDE = []
# If not None, overwrite the KV_KERNEL and Q_KERNEL size with POOL_KVQ_CONV_SIZ.
# Otherwise the kernel_size is [s + 1 if s > 1 else s for s in stride_size].
_C.MVIT.POOL_KVQ_KERNEL = None
# If True, perform no decay on positional embedding and cls embedding.
_C.MVIT.ZERO_DECAY_POS_CLS = True
# If True, use norm after stem.
_C.MVIT.NORM_STEM = False
# If True, perform separate positional embedding.
_C.MVIT.SEP_POS_EMBED = False
# Dropout rate for the MViT backbone.
_C.MVIT.DROPOUT_RATE = 0.0
# If True, use absolute positional embedding.
_C.MVIT.USE_ABS_POS = True
# If True, use relative positional embedding for spatial dimentions
_C.MVIT.REL_POS_SPATIAL = False
# If True, use relative positional embedding for temporal dimentions
_C.MVIT.REL_POS_TEMPORAL = False
# If True, init rel with zero
_C.MVIT.REL_POS_ZERO_INIT = False
# If True, using Residual Pooling connection
_C.MVIT.RESIDUAL_POOLING = False
# Dim mul in qkv linear layers of attention block instead of MLP
_C.MVIT.DIM_MUL_IN_ATT = False
# If True, using separate linear layers for Q, K, V in attention blocks.
_C.MVIT.SEPARATE_QKV = False
# Whether to use the mean pooling of all patch tokens as the output.
_C.MVIT.USE_MEAN_POOLING = False
# If True, use frozen sin cos positional embedding.
_C.MVIT.USE_FIXED_SINCOS_POS = False
# If True, input patches will have overlap
_C.MVIT.PATCH_OVERLAP = False

# This is defines the model size, currently supports Small (S) and Big (B)
_C.MVIT.SIZE = "S"


#------------------------------
# Data options
#------------------------------
_C.DATA = CfgNode()
# Path to the directory which contains data files.
_C.DATA.PATH_TO_DATA_DIR = ""
# If True, we'll use video assets
_C.DATA.USE_VIDEO = True
# If True, we'll use audio assets
_C.DATA.USE_AUDIO = True

# Video
_C.VIDEO = CfgNode()
# Duration of clips in second (during training)
_C.VIDEO.T = 3.0
# Number of frames used after subsampling
_C.VIDEO.N_FRAMES = 16
# Spatial crop size of the input clip.
_C.VIDEO.CROP_SIZE = [224, 224]
# Spatial augmentation jitter scales for training.
_C.VIDEO.CROP_SCALES = [256, 320]

# Audio
_C.AUDIO = CfgNode()
# Size of FFT.
_C.AUDIO.N_FFT = 1024
# Length of hop between STFT windows.
_C.AUDIO.HOP_LENGTH = 501
# Number of mel filterbanks.
_C.AUDIO.N_MELS = 96
# Spatial crop size of the input MelSpectrogram image.
#   when clip length changes, the crop size should be updated accordingly
_C.AUDIO.CROP_SIZE = [96, 288]

#------------------------------
# Misc options
#------------------------------
# Output base directory.
_C.OUTPUT_DIR = "./tmp"

# Note that non-determinism may still be present due to non-deterministic
#   operators under cuda libraries.
_C.RNG_SEED = 1



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_method", default="tcp://localhost:9888", type=str)
    parser.add_argument("--cfg", dest="cfg_file", default=None, type=str)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()

def load_config(args):
    cfg = _C.clone()
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    return cfg
