from fvcore.common.registry import Registry
import math
import torch
import numpy as np
from torchaudio.functional import resample as resample_audio
from torchvision.transforms import Compose, CenterCrop, FiveCrop
from pytorchvideo.transforms import ApplyTransformToKey, UniformTemporalSubsample, Div255, ShortSideScale
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB


# audio sample rate & video fps
SAMPLE_RATE = 48000
FRAMES_PER_SECOND = 25.0

# helper for creating dataset
DATASET_REGISTRY = Registry("DATASET")

def build_dataset(cfg):    
    dataset = DATASET_REGISTRY.get(cfg.TEST.DATASET)(cfg)
    return dataset


# transforms that we apply to audio/video assets at dataloader
# note: this transform class only supports test/inference mode
class Transform(torch.nn.Module):    
    def __init__(self, cfg):
        super().__init__()
        # the clip duration in seconds
        self.T = cfg.TEST.CLIP_LEN
        
        # get the transforms: video
        transform = []
        if cfg.DATA.USE_VIDEO:
            transform.append(
                ApplyTransformToKey(
                    key="video",
                    transform=self.get_video_transforms(cfg)))   
        
        # get the transforms: audio                 
        if cfg.DATA.USE_AUDIO:
            transform.append(
                ApplyTransformToKey(
                    key="audio",
                    transform=self.get_audio_transforms(cfg)))
        
        # combine the transforms into a single unit           
        self.transform = Compose(transform)


    def get_video_transforms(self, cfg):
        transforms = []
        # generate fix number of frames
        transforms.append(
            UniformTemporalSubsample(cfg.VIDEO.N_FRAMES))
        # normalize pixel values in [0,1)
        transforms.append(Div255())
        # resize
        transforms.append(
            ShortSideScale(cfg.VIDEO.CROP_SCALES[0]))
        # crop
        transforms.append(
            FiveCrop(cfg.VIDEO.CROP_SIZE))

        return Compose(transforms)


    def get_audio_transforms(self, cfg):
        transforms = []
        # fixed length + normalization
        transforms.append(
            AudioPreprocess(
                max_len=int(self.T * SAMPLE_RATE),
                do_normalize=False))
        # waveform to spectrogram
        transforms.append(
            MelSpectrogram(
                sample_rate=SAMPLE_RATE, n_fft=cfg.AUDIO.N_FFT,
                hop_length=cfg.AUDIO.HOP_LENGTH, n_mels=cfg.AUDIO.N_MELS)
                )
        # converting magnitude
        transforms.append(AmplitudeToDB())
        # crop
        #   note: crop_size should adapt to T since the width of mel-spec grows with T
        transforms.append(
            CenterCrop(cfg.AUDIO.CROP_SIZE)) 
        
        return Compose(transforms)


    def resample(self, x, fps=FRAMES_PER_SECOND, sample_rate=SAMPLE_RATE):
        if 'video' in x and x['fps'] != fps:
            T = x['video'].shape[1] / x['fps']
            idx = np.linspace(
                start=0,
                stop=x['video'].shape[1] - 1,
                num=math.ceil(T * fps),
                dtype='int',
                )
            x['video'] = x['video'][:,idx]
            x['fps'] = fps

        if 'audio' in x and x['sample_rate'] != sample_rate:
            x['audio'] = resample_audio(
                x['audio'],
                orig_freq=x['sample_rate'],
                new_freq=sample_rate)

    @torch.no_grad()
    def forward(self, x):
        # note: all operations are in-place        
        # fixing fps and sample rate
        self.resample(x)
        # applying preprocessing + augmentation
        self.transform(x)



# helper class to preprocess audio
class AudioPreprocess:
    def __init__(self, max_len, do_normalize=False):
        self.max_len = max_len
        self.do_normalize = do_normalize

    def __call__(self, x):
        assert x.ndim == 2
        # normalization
        if self.do_normalize:
            x = (x - x.mean(dim=1, keepdims=True)) / (x.var(dim=1, keepdims=True) + 1e-5).sqrt()
        # padding
        x = x[:, :self.max_len]
        pad = (0, self.max_len - x.shape[-1])
        x = torch.nn.functional.pad(x, pad, mode='constant', value=0.0)
        return x
