# CCStereo: Audio-Visual Contextual and Contrastive Learning for Binaural Audio Generation

[![Paper](https://img.shields.io/badge/ACM%20MM-2025-blue)](https://doi.org/10.1145/nnnnnnn.nnnnnnn)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)



> Official implementation of **CCStereo: Audio-Visual Contextual and Contrastive Learning for Binaural Audio Generation**  
> [Yuanhong Chen](https://scholar.google.com/citations?user=PiWKAx0AAAAJ&hl=en&oi=ao), [Kazuki Shimada](https://scholar.google.com/citations?user=-t9IslAAAAAJ&hl=en&oi=ao), [Christian Simon](https://scholar.google.com/citations?user=eZrRbp4AAAAJ&hl=en&oi=ao), [Yukara Ikemiya](https://scholar.google.com/citations?user=tWE8kNIAAAAJ&hl=en&oi=ao), [Takashi Shibuya](https://scholar.google.com/citations?user=XCRO260AAAAJ&hl=en&oi=ao), [Yuki Mitsufuji](https://scholar.google.com/citations?user=GMytI10AAAAJ&hl=en&oi=ao)  
> *ACM MM 2025 ([arXiv 2501.02786](https://arxiv.org/abs/2501.02786))*

## Overview
🎧 Binaural Audio Generation (BAG)
CCStereo tackles the task of generating spatialised binaural audio from monaural audio using corresponding visual cues, enabling immersive sound experiences for applications in VR, AR, and 360° video.

🧠 Context-Aware Audio-Visual Conditioning
Existing BAG methods rely heavily on cross-attention mechanisms and often fail to leverage the rich temporal and spatial dynamics present in video. CCStereo introduces Audio-Visual Adaptive De-normalisation (AVAD) layers to modulate the decoding process with spatial and semantic information derived from video frames, offering finer-grained control.

🔍 Spatial-Aware Contrastive Learning
To improve spatial sensitivity, CCStereo uses a novel contrastive learning framework that mines hard negatives by applying spatial shuffling and temporal frame sampling, effectively simulating object position changes and encouraging the model to distinguish fine-grained spatial relationships in the audio-visual space.

🧪 Test-Time Augmentation without Overhead
Unlike prior methods that ignore the inherent redundancy of video data, CCStereo introduces Test-time Dynamic Scene Simulation (TDSS)—a sliding-window based augmentation strategy that crops frames from multiple regions (top-left, center, etc.) without increasing inference cost, boosting robustness and spatial accuracy.

## Highlights

**CCStereo** converts monaural audio to binaural audio using visual input. It addresses key limitations in spatial alignment and generalisation using:

- **AVAD**: Audio-Visual Adaptive De-normalisation for feature modulation.
- **SCL**: Spatial-aware Contrastive Learning for learning spatial correspondence.
- **TDSS**: Test-time Dynamic Scene Simulation for augmentation without added cost.

## Requirements

```bash
pip install -r requirements.txt
```


## Dataset
For training and evaluation, we use the [FairPlay](https://github.com/facebookresearch/FAIR-Play), [YouTube-360](https://github.com/pedro-morgado/spatialaudiogen) datasets. The dataset structure is as follows:
```text
dataset
    ├── fairplay
    ├── yt_clean
    ├── ...
```
You can download the datasets from the links above and place them in the `dataset` directory.


## Training
```bash
bash run_x-your-dataset-name-x.sh
``` 

## Evaluation
The pre-trained ckpt for FairPlay-5Split (split2) can be downloaded from [here](https://drive.google.com/file/d/1CqeV80mt0pZGGH3AN2dZMEv8hOj4mjyv/view?usp=sharing).
```bash
bash python test.py --dataset fairplay ...
```

## License and Citation
This project is licensed under the MIT License. Please see the [LICENSE](LICENSE) file for details.
```
@article{chen2025ccstereo,
  title={Ccstereo: Audio-visual contextual and contrastive learning for binaural audio generation},
  author={Chen, Yuanhong and Shimada, Kazuki and Simon, Christian and Ikemiya, Yukara and Shibuya, Takashi and Mitsufuji, Yuki},
  journal={arXiv preprint arXiv:2501.02786},
  year={2025}
}
```

## Acknowledgements
We acknowledge the use of the [FAIR-Play](https://github.com/facebookresearch/FAIR-Play), [SPADE](https://github.com/NVlabs/SPADE) and [YouTube-360](https://github.com/pedro-morgado/spatialaudiogen). Special thanks to the authors of these works for their contributions to the field of audio-visual learning.

