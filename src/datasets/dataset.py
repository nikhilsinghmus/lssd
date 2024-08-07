import glob
import torch
import numpy as np
from collections import defaultdict
import bisect
from torchvision.io import read_video, read_video_timestamps
from torchaudio import load as read_audio
import torchaudio
from tqdm import tqdm
from .build import DATASET_REGISTRY, Transform



class TimestampsDataset(object):
    def __init__(self, filepath, mtype, T):
        self.filepath = filepath
        self.mtype = mtype
        self.T = T

    def collate_fn(self, x):
        return x

    def __len__(self):
        return len(self.filepath)

    def __getitem__(self, idx):
        pth = self.filepath[idx]

        if self.mtype == "video":
            ts, fps = read_video_timestamps(pth, pts_unit='sec')
            # skip if there was an issue with opening the file
            if len(ts) < 2:
                return None
            x, t = [], 0
            while t + self.T < ts[-1]:
                x.append((float(t), float(t + self.T)))
                t = t + self.T
            if not x:
                # video is too short
                x.append((0.0, float(ts[-1])))
            else:
                # check if more than T/2 sec is left out
                if (float(ts[-1]) - x[-1][1]) > (self.T / 2):
                    x.append((float(ts[-1]) - self.T, float(ts[-1])))

            return (x, pth, fps)

        elif self.mtype == "audio":
            m = torchaudio.info(pth)
            # skip if there was an issue with opening the file
            if m.num_frames < 2:
                return None
            x, i = [], 0
            while (i + 1) * self.T * m.sample_rate < m.num_frames:
                x.append((int(i * self.T * m.sample_rate), int((i + 1) * self.T * m.sample_rate)))
                i = i + 1
            if not x:
                # audio is too short
                x.append((0, m.num_frames))
            else:
                # check if more than T/2 sec is left out
                if m.num_frames - x[-1][1] > int(self.T * m.sample_rate / 2):
                    x.append((m.num_frames - int(self.T * m.sample_rate), m.num_frames))

            return (x, pth, m.sample_rate)

        else:
            raise NotImplementedError

    def run(self, cfg):
        items_dict = defaultdict(list)

        dl = torch.utils.data.DataLoader(
            self,
            batch_size=cfg.TEST.BATCH_SIZE, 
            num_workers=cfg.DIST.NUM_WORKERS,
            collate_fn=self.collate_fn,
            )

        for data in tqdm(dl, desc=f'Reading Timestamps: {self.mtype}', dynamic_ncols=True):
            # discard cases where opening the file has not been successful
            data = list(filter(lambda x: x, data))
            ranges, pths, rates = list(zip(*data))
            # keep track of metadata
            items_dict['range'].extend(ranges)
            items_dict['filepath'].extend(pths)
            items_dict['sample_rate'].extend(rates)

        cumulative_sizes = np.cumsum(
            list(map(len, items_dict['range']))) if 'range' in items_dict else [0]

        return items_dict, cumulative_sizes



@DATASET_REGISTRY.register()
class DatasetFolder(object):
    # these are the extensions we support by default
    MEDIA_TYPE_EXTN = {
        'video': ('mp4','avi'), 
        'audio': ('mp3','wav','flac', 'opus')
        }
    def __init__(self, cfg):
        assert torch.distributed.is_initialized() 
        for mtype, extns in self.MEDIA_TYPE_EXTN.items():
            dataset = Dataset(cfg, mtype, extns)
            if len(dataset) > 0:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset=dataset,
                    shuffle=False,
                    seed=cfg.RNG_SEED,
                    drop_last=False,
                    )

                loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=cfg.TEST.BATCH_SIZE,
                    shuffle=False, # it is applied through sampler
                    sampler=sampler,
                    num_workers=cfg.DIST.NUM_WORKERS,
                    collate_fn=None,
                    pin_memory=True,
                    )

                setattr(self, f'{mtype}_loader', loader)



class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, mtype, extns):
        # clone since we might need to modify the config on-the-fly
        self.cfg = cfg.clone()
        # we process one modality at a time
        self.mtype = mtype
        if mtype == 'video':
            assert self.cfg.DATA.USE_VIDEO
            self.cfg.DATA.USE_AUDIO = False
        elif mtype == 'audio':
            assert self.cfg.DATA.USE_AUDIO
            self.cfg.DATA.USE_VIDEO = False
        else:
            NotImplementedError
        # clip duration for inference
        T = cfg.TEST.CLIP_LEN
        # get the path to media files
        files = []
        for ext in extns:
            files.extend(glob.glob(f"{cfg.DATA.PATH_TO_DATA_DIR}/**/*.{ext}", recursive=True))
        # preparing the list of media chunks which need to be processed
        self.items_dict, self.cumulative_sizes = TimestampsDataset(files, mtype, T).run(cfg)
        # data transforms
        self.transform = Transform(self.cfg)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def unfltn_indx(self, idx):
        vidx = bisect.bisect_right(self.cumulative_sizes, idx)
        cidx = idx if vidx == 0 else idx - self.cumulative_sizes[vidx - 1]
        return vidx, cidx

    def read(self, idx):
        # convert flat global index to video and clip indices
        vidx, cidx = self.unfltn_indx(idx)

        filepath = self.items_dict['filepath'][vidx]
        start_pts, end_pts = self.items_dict['range'][vidx][cidx]

        if self.mtype == 'video':
            x, _, fps = read_video(filepath, start_pts, end_pts, pts_unit='sec')
            x = x.permute(3, 0, 1, 2) # (C, T, H, W)
            return {'video': x, 'idx': idx, 'fps':fps['video_fps']}

        elif self.mtype == 'audio':
            x, sample_rate = read_audio(filepath, start_pts, end_pts - start_pts) # (*, S)
            if x.shape[0] > 1:
                assert x.shape[0] in (1,2) # currently only supporting mono and stereo
                x = torch.mean(x, dim=0, keepdim=True)
            return {'audio': x, 'idx': idx, 'sample_rate':sample_rate}

        else:
            raise NotImplementedError


    def __getitem__(self, idx):
        # read raw media assets
        x = self.read(idx)
        # apply test-time transforms (operator is in-place)
        self.transform(x)
        return x
