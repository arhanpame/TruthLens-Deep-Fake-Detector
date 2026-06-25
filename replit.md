# DeepFake Detector

A web app that classifies an image as **Real** or **AI-Generated** using a Keras CNN. Upload an image or pick a sample, and the app shows the predicted label, a confidence score, and per-class probability bars.

## Run & Operate

- API server (Python FastAPI) and frontend (React/Vite) run as Replit workflows — do not start them manually with `pnpm dev`.
  - `artifacts/api-server: API Server` → FastAPI, serves `/api/*`
  - `artifacts/detector: web` → static React frontend, served at `/`
- `pnpm run typecheck` — full typecheck across all packages
- Python deps: `tensorflow-cpu`, `fastapi`, `uvicorn`, `python-multipart`, `numpy`, `pillow` (managed via the package tools)

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- **Backend: Python + FastAPI** (`deepfake/server.py`), run with `uvicorn`
- **ML: TensorFlow / Keras** — `deepfake/deepfake_detector.keras`
- Frontend: React + Vite (static), hand-written `App.tsx` calling `/api/*` via `fetch`

## Where things live

- `deepfake/server.py` — FastAPI backend (authoritative). Endpoints: `GET /api/healthz`, `GET /api/samples`, `GET /api/samples/{name}`, `POST /api/predict`, `POST /api/predict-sample/{name}`.
- `deepfake/helpers.py` — `predict(model, image, class_names)` (resize 128×128, /255 normalize, single sigmoid → P(AI-Generated)).
- `deepfake/model_setup.py` — `paths` dict (files only) + `dir`; `model_setup.dir` is the base for the `sample_images/` directory.
- `deepfake/class_names.json` — `["Real", "AI-Generated"]`.
- `deepfake/sample_images/` — preloaded `AI-Generated_*.png` and `Real_*.png` samples.
- `artifacts/detector/src/App.tsx` + `src/index.css` — frontend UI and theme.
- `artifacts/api-server/.replit-artifact/artifact.toml` — backend run config (dev + prod uvicorn).

## Architecture decisions

- Streamlit was abandoned: the pnpm artifact template can't make Streamlit previewable/deployable. Architecture is a static React frontend at `/` + Python FastAPI backend at `/api`, routed by the shared application proxy.
- The `api-server` artifact's Node/Express scaffold (`src/`) is dead code — the artifact runs `uvicorn` via `artifact.toml`, not the Node `dev` script.
- The artifact's run command executes with cwd = the artifact directory, so `artifact.toml` uses `--app-dir ../../deepfake` to reach the repo-root model files.
- Model loads lazily on the first prediction (~2s first inference, then fast). `/api/healthz` reports `model_loaded`.
- Ground-truth label for samples is inferred from filename prefix (`AI-Generated_*` vs `Real_*`); no DB is used.

## Product

Single-page tool: drop/upload an image or click a preloaded sample. The backend runs the Keras model and returns the label (threshold 0.5 on P(AI-Generated)), confidence, raw probability, and inference time, rendered as a verdict card plus Real / AI-Generated probability bars.

## User preferences

_Populate as you build — explicit user instructions worth remembering across sessions._

## Gotchas

_Populate as you build — sharp edges, "always run X before Y" rules._

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
