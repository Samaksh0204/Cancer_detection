# Oral Cancer Detection using CNNs

Four-class classification of oral cavity photographs (Healthy / Benign / OPMD / OCA) with
transfer learning (MobileNetV2, DenseNet121 and an ensemble of both). The models also learn to
point at the lesion, take the patient's risk factors (age, gender, smoking, betel quid, alcohol)
as a second input, and are read as a screening tool: *does this patient need a specialist?*

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

5-fold cross-validation by patient (mean ± standard deviation over the folds; every photo is
tested exactly once):

| Model | Accuracy | Macro-F1 | OCA recall | Referral AUC | OCA AUC |
|---|---|---|---|---|---|
| MobileNetV2 | 60.1% ± 5.3 | 0.58 ± 0.07 | 57% ± 20 | 0.79 ± 0.05 | 0.91 ± 0.04 |
| DenseNet121 | 61.5% ± 4.5 | 0.57 ± 0.06 | 54% ± 21 | 0.81 ± 0.03 | 0.89 ± 0.01 |
| Ensemble of both | 62.1% ± 4.5 | 0.59 ± 0.07 | 56% ± 21 | 0.81 ± 0.04 | 0.93 ± 0.03 |

Referral AUC = telling OPMD/OCA apart from Healthy/Benign; OCA AUC = telling cancer apart from
everything else (1.0 is perfect, 0.5 is chance). Always predicting the largest class (OPMD) gives
46.5% accuracy.

Single held-out test split (600 images, 26 of them OCA), ensemble: 58.5% accuracy, 0.58 macro-F1,
62% OCA recall. Used as a screening tool, with thresholds chosen on the validation set to catch
about 90% of cases:

| Question | Sensitivity | Specificity | AUC |
|---|---|---|---|
| Referral (OPMD or OCA) | 80% | 58% | 0.78 |
| Suspected cancer (OCA) | 92% | 67% | 0.92 |

The referral threshold caught fewer cases on the test set (80%) than the 90% it was tuned for on
validation, and the folds differ a lot in ensemble OCA recall (31% to 81%): with only 129 cancer photos
these numbers are still noisy.

What helped, compared with plain fine-tuning (ensemble, cross-validated: 58.8% accuracy, 0.55
macro-F1): the lesion outlines and the patient risk factors described under Notes. What did not
help: more dropout / weight decay / label smoothing, 384-pixel inputs, and larger backbones
(EfficientNetV2-S, ConvNeXt-Tiny). The limit appears to be the data (about 1,900 training
photos) rather than the training recipe.

## Training on a GPU

TensorFlow >= 2.11 cannot use the GPU on native Windows. If PyTorch with CUDA is installed, the
notebook runs Keras on PyTorch instead and uses the GPU (about 6x faster on an RTX 3060):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Without it the notebook runs on TensorFlow on the CPU, and the cross-validation section is skipped
(set `ORAL_CV=1` to run it anyway; it takes many hours on a CPU).

## Notes

- Each photo is rotated upright (EXIF) and cropped to the oral-cavity box from
  `Annotation.json` before resizing; the whole backbone is then fine-tuned with fairly strong
  augmentation.

- Lesion guidance: the lesion outlines in `Annotation.json` are used as a second training target.
  Each model predicts a coarse 7x7 map of where the lesion is and pools its features from that
  area, so it has to look at the lesion rather than the background. The outlines are only needed
  for training, not for prediction.

- Risk factors: age, gender, smoking, betel quid chewing and alcohol from the patient table are
  fed to the classifier next to the image features. `predict_full(image_path, tobacco, betel,
  alcohol, age, gender)` takes the same details for a new photo, prints the class probabilities
  and the two screening flags, and shows the predicted lesion area.

- Train/validation/test are split by patient, and class imbalance is handled with class
  weights on the training set, so the reported test scores are free of leakage. Models and
  screening thresholds are chosen on the validation set only.
