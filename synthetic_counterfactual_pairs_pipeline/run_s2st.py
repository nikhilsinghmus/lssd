import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "lvc"))
import glob
import math
import torch
import torchaudio
import hydra
import whisper
import pyloudnorm
from tqdm.auto import tqdm
from omegaconf import DictConfig, OmegaConf
from seamless_communication.models.inference import Translator
from lvc.inference_wav2vec import LVC_VC_Inference
from typing import Union


TMPDIR = os.environ.get("TMPDIR", "/tmp")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_s2st(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    files = glob.glob(
        os.path.join(
            cfg.paths.current_path,
            "separated/mdx_extra/**",
            "vocals.wav"
        ),
        recursive=True
    )

    files = sorted(files)
    total_files = len(files)

    # Subset files if needed
    if cfg.s2st.subset.from_idx > 0:
        files = files[cfg.s2st.subset.from_idx:]
    if cfg.s2st.subset.to_idx > 0:
        files = files[:(cfg.s2st.subset.to_idx - cfg.s2st.subset.from_idx)]

    print(
        "Will process %d of %d files, from %d to %d." % (
            len(files),
            total_files,
            cfg.s2st.subset.from_idx,
            cfg.s2st.subset.to_idx
        )
    )

    # Load models
    recognizer = whisper.load_model("large").cuda()

    translator = Translator(
        "seamlessM4T_large",
        vocoder_name_or_card="vocoder_36langs",
        device=torch.device("cuda:0"),
        dtype=torch.float16
    )

    vc_hp = OmegaConf.load(cfg.vc.config)
    vc = LVC_VC_Inference(
        vc_hp,
        lvc_vc_chkpt=cfg.vc.lvc_vc_weights,
        speaker_encoder_chkpt=cfg.vc.se_weights,
        seen_speaker_emb_gmms_pkl=cfg.vc.seen_speaker_emb_gmms_pkl,
        seen_speaker_f0_metadata_pkl=cfg.vc.seen_speaker_f0_metadata_pkl,
        device="cuda:0"
    )

    for file in tqdm(files):
        audio, sr, segments = make_segments(file, recognizer) # Extract segments
        for language in cfg.s2st.languages:
            outf = os.path.join(
                os.path.dirname(file),
                "%s-%s.wav" % (os.path.splitext(os.path.basename(file))[0], language)
            )

            processed_audio, outsr = process_audio_in_segments( # Process segments
                audio,
                16000, # Always 16k
                translator,
                vc,
                segments,
                cfg.s2st.strategy,
                language
            )

            if outsr != sr:
                processed_audio = torchaudio.functional.resample(
                    processed_audio,
                    outsr,
                    sr
                )

            torchaudio.save(outf, processed_audio, sr)


@torch.inference_mode()
def make_segments(file: str, recognizer: whisper.model.Whisper) -> tuple[torch.Tensor, int, list[dict[str, Union[str, float]]]]:
    print("Processing %s." % file)
    f_16k = os.path.join(
        os.path.dirname(file),
        os.path.splitext(os.path.basename(file))[0] + "_16k.wav"
    )
    y, sr = torchaudio.load(file, normalize=False)
    y = y.to(dtype=torch.float32)
    y /= (2 ** 15)
    if sr != 16000: # Always resample to 16k
        y = torchaudio.functional.resample(y, sr, 16000)
    torchaudio.save(f_16k, y, 16000)

    segments = recognizer.transcribe(
        file,
        verbose=True,
        word_timestamps=True,
        language="en"
    )["segments"] # Extract segments

    return y, sr, segments


@torch.inference_mode()
def process_audio_in_segments(
    audio: torch.Tensor,
    sample_rate: int,
    model: Translator,
    vc: LVC_VC_Inference,
    segments: list[dict[str, Union[str, float]]],
    mode: str,
    language: str
) -> tuple[torch.Tensor, int, int]:
    meter = pyloudnorm.Meter(16000) # Always 16k

    # For the loudness normalization
    loudness_ref = meter.integrated_loudness(audio.T.cpu().numpy()).T

    output_audio = torch.zeros_like(audio.mean(dim=0, keepdim=True)) # Empty buffer
    for segment in tqdm(segments):
        start = math.floor(segment["start"] * sample_rate)
        end = math.ceil(segment["end"] * sample_rate)
        duration = end - start

        if (duration < (sample_rate * 0.5)): # Skip any segments <0.5s
            continue

        segment_audio = audio[:, start:end] # Extract segment

        segment_tmpfile = os.path.join(TMPDIR, "segment.wav")
        torchaudio.save(segment_tmpfile, segment_audio, sample_rate)

        _, wav_segment, outsr_segment = model.predict( # Translate segment
            segment["text"] if mode == "t2st" else segment_tmpfile, # Use text if mode is t2st
            mode,
            language,
            "eng"
        )

        # Check if the sample rate of the output is the same as the input
        assert outsr_segment == sample_rate, "Sample rate mismatch: %d for output vs %d for input." % (outsr_segment, sample_rate)

        wav_segment = wav_segment.to(dtype=torch.float32).cpu()[0] # Convert to float32 and remove the batch dimension

        # Compute the stretch ratio
        ratio = wav_segment.shape[1] / duration
        ratio = min(max(ratio, 0.5), 2)

        # Apply the tempo effect
        wav_segment = torchaudio.sox_effects.apply_effects_tensor(
            wav_segment,
            sample_rate,
            [
                ["tempo", "-s", str(ratio)]
            ]
        )[0]

        # Run the voice conversion
        wav_segment = vc.run_inference(
            source_audio=wav_segment.numpy().squeeze(),
            target_audio=segment_audio.mean(dim=0).numpy(),
            source_seen=False,
            target_seen=False,
            source_id=None,
            target_id=None
        )
        wav_segment = torch.from_numpy(wav_segment)[None, :]

        new_end = min(start + wav_segment.shape[1], output_audio.shape[1])
        output_audio[:, start:new_end] = wav_segment[:, :min(new_end - start, wav_segment.shape[1])]

    # Loudness normalization
    loudness = meter.integrated_loudness(output_audio.T.cpu().numpy())
    output_audio = pyloudnorm.normalize.loudness(output_audio.T.cpu().numpy(), loudness, loudness_ref).T
    output_audio = torch.from_numpy(output_audio).to(dtype=torch.float32)

    return output_audio, sample_rate


if __name__ == "__main__":
    run_s2st()
