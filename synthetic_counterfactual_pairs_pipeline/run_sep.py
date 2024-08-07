import os
import glob
import hydra
import multiprocessing
import demucs.separate
from tqdm.auto import tqdm
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_sep(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    files = []
    for ext in cfg.extensions:
        files.extend(
            glob.glob(
                os.path.join(
                    cfg.dataset.paths.video,
                    "**/*." + ext
                ),
                recursive=True
            )
        )

    if cfg.n_processes > 1:
        with multiprocessing.Pool(cfg.n_processes) as pool: # Run sep_audio in parallel
            list(tqdm(pool.imap(sep_audio, files), total=len(files)))
    else:
        for f in tqdm(files):
            sep_audio(f)


def sep_audio(f: str) -> None:
    # Run demucs in two stems mode
    demucs.separate.main(["--two-stems", "vocals", "-n", "mdx_extra", f])


if __name__ == "__main__":
    run_sep()
