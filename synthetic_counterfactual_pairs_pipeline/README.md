# Looking Similar Sounding Different: Leveraging Counterfactual Cross-Modal Pairs for Audiovisual Representation Learning
#### CVPR 2024

Synthetic Counterfactual Pairs Pipeline

## Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@inproceedings{singh2024looking,
  title={Looking similar sounding different: Leveraging counterfactual cross-modal pairs for audiovisual representation learning},
  author={Singh, Nikhil and Wu, Chih-Wei and Orife, Iroro and Kalayeh, Mahdi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={26907--26918},
  year={2024}
}
```

# Setup
This portion of the code has separate requirements.

```bash
pip install -r requirements.txt
```

In addition, you'll need a copy of [LVC-VC](https://github.com/wonjune-kang/lvc-vc) [1]. You will need their pretrained weights, which should be deposited in `./lvc/weights`.

```bash
git clone https://github.com/wonjune-kang/lvc-vc lvc
```

[1] Kang, Wonjune, Mark Hasegawa-Johnson, and Deb Roy. "End-to-End Zero-Shot Voice Conversion with Location-Variable Convolutions." arXiv preprint arXiv:2205.09784 (2022).

## Inference

The inference pipeline consists of several sequential steps, each handled by a separate script. Here's a detailed overview of the process:

1. **Extract Audio**
   ```bash
   python3 extract_audio.py
   ```
   Demuxes the input videos (separates the audio and video streams).

2. **Run Source Separation**
   ```bash
   python3 run_sep.py
   ```
   Uses [Demucs](https://github.com/facebookresearch/demucs) to extract vocal stems from the audio.

3. **Run Speech/Transcript-to-Speech Translation**
   ```bash
   python3 run_s2st.py
   ```
   Performs speech/transcript-to-speech translation on the extracted vocal stems.

4. **Remix and Remux**
   ```bash
   python3 remix_nosr.py
   ```
   Remixes the translated speech with the original background audio track and remuxes it with the video.

### Configuration

We use [Hydra](https://github.com/facebookresearch/hydra) for configuration management. The main configuration file is `conf/config.yaml`. You can override configuration parameters from the command line.

Key configuration sections:

- `dataset`: Contains data paths and related settings.
- `s2st`: Configures the speech-to-speech translation process (currently uses [Seamless M4T](https://github.com/facebookresearch/seamless_communication)).
- `model`: Specifies and configures the source separation model (default: Demucs).
- `vc`: Voice conversion settings. Right now, we only support LVC, but this is structured so as to allow a user specified alternative.

For example:

Process a subset of the dataset (e.g. for launching parallel jobs on a cluster).
   ```bash
   python3 run_s2st.py s2st.subset.from_idx=100 s2st.subset.to_idx=200
   ```

Change the translation strategy:
   ```bash
   python3 run_s2st.py s2st.strategy=t2st
   ```
