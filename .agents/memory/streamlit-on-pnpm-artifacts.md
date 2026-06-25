---
name: Python ML model app on the pnpm artifact template
description: How to ship a Python/Keras model as a previewable+deployable app when the only template is the pnpm monorepo (no Streamlit artifact kind)
---

# Python ML model app on the pnpm artifact template

When a user brings a Python ML model (Keras/TF, sklearn, etc.) and wants a previewable/deployable web app, **do not use Streamlit**. The pnpm artifact template has no Streamlit artifact kind, an artifact's `kind` can't be changed after creation, and the `web` kind serves static files only — so a Streamlit process can't be previewed or deployed as an artifact.

**Working architecture:** static React frontend artifact at `/` + Python backend artifact (reuse the `api` artifact) serving `/api`. The frontend calls `/api/*` with `fetch`; the shared proxy routes by path.

**Why:** the global reverse proxy routes by path from each artifact's `artifact.toml`; a static `web` artifact + a separate backend service on `/api` is the supported way to combine a JS UI with a non-Node runtime.

## Critical gotchas (cost real debugging time)

- **The `api` artifact's run command comes from `artifact.toml`'s `[services.development] run` / `[services.production.run]`, NOT the package.json `dev` script.** Editing the toml then restarting the workflow swaps the command. Editing package.json `dev` does nothing for the artifact-managed workflow. Use `verifyAndReplaceArtifactToml` to change it.
- **The run command executes with cwd = the artifact directory** (e.g. `artifacts/api-server/`). To reach files elsewhere in the repo (model at repo-root `deepfake/`), use a relative path from there: `uvicorn server:app --app-dir ../../deepfake ...`. Set the SAME path in both development and production blocks.
- After changing `artifact.toml`, the **running workflow keeps the old command until restarted** — and the captured `/tmp/logs/*.log` file can be stale. Use `getWorkflowStatus({name})` to read the live command + output and confirm the new command took effect.
- `model_setup.paths` (the auto-generated helper) maps **files only**, not directories. For a samples/assets *directory*, build it from `model_setup.dir` (e.g. `os.path.join(model_setup.dir, "sample_images")`), don't expect a dict key for it.
- Load the TF/Keras model **lazily on first request**, not at import/startup — first inference is ~2s, and lazy loading keeps the workflow's startup health check fast.
