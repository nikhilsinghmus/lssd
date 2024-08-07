import os
import logging
import shutil
import datetime
import subprocess
import glob
import functools
import multiprocessing
import numpy
import soundfile
import librosa
import hydra
import matchering
import pyloudnorm
from tqdm.auto import tqdm
from omegaconf import DictConfig, OmegaConf


TMPDIR = os.environ.get("TMPDIR", "/tmp")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_remixnosr(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    files = glob.glob(
        os.path.join(
            cfg.paths.current_path,
            "separated/mdx_extra/**",
            "no_vocals.wav"
        ),
        recursive=True
    )

    new_videodir = os.path.join(
        os.path.dirname(cfg.dataset.paths.video),
        "video_%s" % datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    logging.info("Copying over video files.")
    shutil.copytree(
        cfg.dataset.paths.video,
        new_videodir
    )
    cfg.dataset.paths.video = new_videodir

    files = sorted(files)

    logging.info("Running mixing.")
    with multiprocessing.Pool(cfg.n_processes) as pool:
        list(
            tqdm(
                pool.imap(
                    functools.partial(
                        process_file,
                        cfg=cfg
                    ),
                    files
                ),
                total=len(files)
            )
        )


def process_file(file: str, cfg: DictConfig) -> None:
    file_orig = file.replace("no_vocals.wav", "vocals.wav")
    video_id = file.split("/")[-2]
    video_path = os.path.join(cfg.dataset.paths.video, video_id + ".mp4")

    y_background, sr_background = soundfile.read(file)
    y_orig, sr_orig = soundfile.read(file_orig)

    for language in tqdm(cfg.s2st.languages, leave=False):
        file_language = file.replace("no_vocals.wav", "vocals-%s.wav" % language)
        file_resampled = file_language.replace(".wav", "-resampled.wav")
        file_matched = file_resampled.replace(".wav", "-matched.wav")

        y_language, sr_language = soundfile.read(file_language)

        if not sr_language == sr_background:
            y_language = librosa.resample(y_language, orig_sr=sr_language, target_sr=sr_background)
            sr_language = sr_background

        soundfile.write(
            file_resampled,
            y_language,
            sr_language
        )

        matchering.process(
            target=file_resampled,
            reference=file.replace("no_vocals.wav", "vocals.wav"),
            results=[
                matchering.Result(
                    file_matched,
                    subtype="FLOAT",
                    use_limiter=False,
                    normalize=False
                )
            ]
        )

        y_speech, sr_speech = soundfile.read(file_matched)

        if not sr_speech == sr_background:
            y_speech = librosa.resample(y_speech, orig_sr=sr_speech, target_sr=sr_background)
            sr_speech = sr_background

        if len(y_speech.shape) == 1:
            y_speech = numpy.stack([y_speech, y_speech], axis=1)

        y_orig, sr_orig = soundfile.read(file_orig)

        meter = pyloudnorm.Meter(sr_orig)
        loudness_orig = meter.integrated_loudness(y_orig)

        meter = pyloudnorm.Meter(sr_speech)
        loudness_speech = meter.integrated_loudness(y_speech)

        y_speech = pyloudnorm.normalize.loudness(y_speech, loudness_speech, loudness_orig)

        y_speech = y_speech[:len(y_background)]
        y_mix = y_background + y_speech

        y_mix /= numpy.abs(y_mix).max()


        tmp_filename = os.path.join(TMPDIR, "mix-%s.wav" % video_id)
        soundfile.write(
            tmp_filename,
            y_mix,
            sr_background
        )

        subprocess.run(
            [
                "ffmpeg",
                "-i",
                video_path,
                "-i",
                tmp_filename,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-strict",
                "-2",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                os.path.join(cfg.dataset.paths.video, "%s_%s.mp4" % (video_id, language))
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )


if __name__ == "__main__":
    run_remixnosr()
