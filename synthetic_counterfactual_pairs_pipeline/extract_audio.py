import os
import subprocess
import glob
import multiprocessing
import hydra
from tqdm.auto import tqdm
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_audioextract(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    video_files = glob.glob(os.path.join(cfg.dataset.paths.video, "*.mp4"))

    with multiprocessing.Pool(cfg.n_processes) as pool: # Run extract_audio in parallel
        list(tqdm(pool.imap(extract_audio, video_files), total=len(video_files)))


def extract_audio(file: str) -> None:
    audio_file = os.path.splitext(file)[0] + ".wav"
    cmd = ["ffmpeg", "-i", file, "-vn", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "2", audio_file]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    run_audioextract()
