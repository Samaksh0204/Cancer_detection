# Oral Cancer Detection using CNNs

Four-class classification of oral cavity photographs (Healthy / Benign / OPMD / OCA) with
transfer learning (MobileNetV2 vs DenseNet121), plus a simple lifestyle risk score.

> Research/educational project only — not a medical device, not for diagnosis.

## Run

```bash
pip install -r requirements.txt
```

Copy the dataset files as downloaded from Zenodo (`Images.zip`, `Imagewise_data.csv`,
`Patientwise_data.csv`, `Annotation.json`) into the `data/` folder, then run `OralCancerDetection_UsingCNN.ipynb`
top to bottom. The zip is extracted automatically. To keep the dataset elsewhere, set
`ORAL_DATA_DIR` to that folder instead. Trained models are saved to `models/`.

Set `ORAL_SMOKE=1` to run a single epoch per stage as a quick end-to-end check.

Dataset (access on request): https://zenodo.org/records/10664056 — described in Piyarathne et al.,
Oral Oncology 156 (2024) 106946, https://doi.org/10.1016/j.oraloncology.2024.106946

## Results

Held-out test set (600 images from patients never seen in training; 26 are OCA):

| Model | Accuracy | Macro-F1 | OCA recall |
|---|---|---|---|
| MobileNetV2 (selected by validation macro-F1) | 52.3% | 0.51 | 62% |
| DenseNet121 | 51.7% | 0.51 | 50% |

Always predicting the largest class (OPMD) gives 46.5% accuracy. With so few OCA test images,
OCA recall varies a lot between runs.

## Notes

- Each photo is rotated upright (EXIF) and cropped to the oral-cavity box from
  `Annotation.json` before resizing; the whole backbone is then fine-tuned.

- Train/validation/test are split by patient, and class imbalance is handled with class
  weights on the training set, so the reported test scores are free of leakage.
- TensorFlow >= 2.11 has no GPU support on native Windows; training runs on the CPU there.
  Use WSL2 to train on an NVIDIA GPU.
