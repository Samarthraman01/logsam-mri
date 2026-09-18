# LoGSAM — MRI Brain Tumor Segmentation

Reproduction and extension of:
> **LoGSAM: Parameter-Efficient Cross-Modal Grounding for MRI Segmentation**
> Bhuiyan et al., arXiv:2603.17576, 2026

## What This Project Does

Takes a radiologist's spoken dictation and automatically segments brain tumors in MRI scans.

Speech → Text → Tumor Prompt → Bounding Box → Pixel Mask


## Pipeline

1. **Speech module** — Whisper ASR transcribes radiologist dictation
2. **NLP module** — spaCy + negspaCy extracts tumor class and handles negation
3. **Detection module** — LoRA-adapted Grounding DINO localizes tumor with a bounding box
4. **Segmentation module** — MedSAM converts the bounding box into a pixel-level mask

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/logsam-mri.git
cd logsam-mri
conda activate deeplearn
pip install -r requirements.txt
```

## Project Structure

logsam-mri/
├── data/ # datasets (not tracked by git)
├── src/
│ ├── speech/ # Whisper ASR + NLP prompt extraction
│ ├── detection/ # LoRA-augmented Grounding DINO
│ ├── segmentation/ # MedSAM inference
│ └── utils/ # shared helpers
├── notebooks/ # exploration and visualization
├── checkpoints/ # saved model weights (not tracked)
├── results/ # evaluation outputs
└── configs/ # training configuration files


## Results

| Model | Dice Score | mAP@50 | Trainable Params |
|-------|-----------|--------|-----------------|
| Fully fine-tuned GDINO + MedSAM | 0.8145 | 0.8499 | 100% |
| LoGSAM (ours) | 0.8032 | 0.8282 | 4.96% |

## Dataset

- Primary: BRISC 2025
- OOD evaluation: Kaggle MRI Brain Tumor with Bounding Boxes

## Reference

Bhuiyan et al. "LoGSAM: Parameter-Efficient Cross-Modal Grounding for MRI Segmentation." arXiv:2603.17576 (2026).
