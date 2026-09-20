[README.md](https://github.com/user-attachments/files/32429733/README.md)
# TruthLens — AI-Generated Image Detector

A convolutional neural network that classifies images as **real photographs** or **AI-generated**, served through a web app with confidence scores and click-to-try examples.

Built during the Inspirit AI summer program. The models were trained in Google Colab; the web interface was scaffolded with Replit Agent. See [Attribution](#attribution) for the full breakdown.

---

## The problem

Modern text-to-image models produce images that are difficult to identify as synthetic by eye. This project trains a classifier to make that call, and — more interestingly — examines *where it fails*.

## Dataset

A subset of **GenImage**, balanced between real photographs and AI-generated images drawn from four different generators:

| Split | Real | AI-generated | Total |
|---|---|---|---|
| Train | 5,600 | 5,600 | 11,200 |
| Test  | 1,400 | 1,400 | 2,800 |

Generators represented: **Stable Diffusion 2.1, SDXL, Stable Diffusion 3, DALL·E 3**.

All images are resized to 128×128 RGB and normalized to `[0, 1]`.

## Models

### 1. CNN trained from scratch

```
Input(128, 128, 3)
Conv2D(32, 3x3, relu) → MaxPooling2D
Conv2D(64, 3x3, relu) → MaxPooling2D
Conv2D(128, 3x3, relu) → MaxPooling2D
GlobalAveragePooling2D
Dense(128, relu)
Dropout(0.5)
Dense(1, sigmoid)
```

109,889 parameters · Adam (lr = 1e-3) · binary crossentropy · EarlyStopping on validation loss (patience 3, best weights restored)

**Test accuracy: 82.0%** — 280 false positives, 223 false negatives out of 2,800 test images.

### 2. Transfer learning — Xception

ImageNet-pretrained **Xception** backbone with the final 20 layers unfrozen. 21,123,881 parameters.

**Test accuracy: 85.4%** — a 3.4 point improvement over the scratch CNN. This is the model deployed in the app.

## What went wrong, and what I learned from it

The transfer model **overfit sharply**. Over seven epochs, training accuracy climbed 0.81 → 0.99 while validation loss moved in the wrong direction:

| Epoch | Train acc | Val loss |
|---|---|---|
| 1 | 0.807 | 0.419 |
| 2 | 0.901 | **0.390** ← best |
| 3 | 0.949 | 0.603 |
| 4 | 0.969 | 0.734 |
| 5 | 0.977 | 0.728 |
| 6 | 0.986 | 0.796 |
| 7 | 0.991 | 0.708 |

The model was memorizing the training set rather than learning generalizable artifacts. EarlyStopping caught it and restored the epoch-2 weights, which is why the reported figure is 85.4% rather than the higher validation accuracy that appeared briefly in later epochs.

The likely cause is the fine-tuning learning rate. I used **1e-3**, the same rate as the scratch CNN, where fine-tuning a pretrained backbone generally calls for something closer to **1e-4**. Too large a step size on pretrained weights erodes the ImageNet features the backbone was chosen for in the first place.

## Known limitations

- **The reported accuracy is optimistic.** The same held-out set was used both to trigger EarlyStopping and to report final accuracy, so it is not a clean estimate of unseen-data performance. A separate validation split would fix this.
- **No cross-generator holdout.** All four generators appear in both train and test. The harder and more realistic question — does a detector trained on SD and DALL·E generalize to a generator it has never seen — is untested here.
- **Compression artifacts are a confound.** Real and generated images may differ in JPEG history, so some of the signal may come from compression rather than generation.
- Input is downsampled to 128×128, which discards fine high-frequency detail that is often where generation artifacts live.

## Analysis included in the notebook

- Per-generator accuracy breakdown (which generators evade detection most successfully)
- Confusion matrix with false-positive / false-negative tradeoff
- Misclassified example galleries
- **Gradient-based saliency maps** — visualizing which image regions drive the prediction, used to check whether the model attends to plausible artifacts or to spurious background cues

## Application

Decoupled frontend/backend:

- **Backend** — Python **FastAPI** at `/api/*`, loading the Keras model lazily (~2s on first inference, cached after)
- **Frontend** — **React + Vite** (TypeScript) static site
- **Orchestration** — pnpm workspaces, Node.js 24

Inference path: image → RGB → resize 128×128 → divide by 255 → batch dimension → sigmoid. Output ≥ 0.5 is labeled `AI-Generated`; `P(Real) = 1 − sigmoid`. (Single-output sigmoid — thresholded, not argmaxed.)

## Repository layout

```
deepfake/
  deepfake_detector.keras     trained Xception transfer model
  class_names.json            ["Real", "AI-Generated"]
  helpers.py                  preprocessing + prediction
  server.py                   FastAPI inference endpoints
  model_setup.py              model file path resolution
  sample_images/              preloaded examples
lib/, scripts/                frontend and build tooling
main.py                       entry point
Copy_of_Student_TruthLens_Section2.ipynb    training notebook
```

## Running locally

```bash
pip install -r deepfake/requirements.txt
python main.py
```

## Attribution

Being precise about this, since it matters:

- **Mine** — model architecture choices, training and hyperparameters, backbone selection, error analysis, and the diagnosis of the overfitting behavior described above. All of this lives in the training notebook.
- **Replit Agent** — the React frontend and FastAPI serving layer built around the trained model. This is why the repository's commit history and language statistics are dominated by generated TypeScript.

## Next steps

- Retrain the transfer model at lr = 1e-4 and compare against the 85.4% baseline
- Introduce a proper validation split, separate from the test set
- Hold out one generator entirely at training time to measure cross-generator generalization
