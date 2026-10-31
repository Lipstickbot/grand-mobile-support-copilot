# Local Qwen3-8B Soup smoke experiment

This experiment is separate from the unfinished application baseline. User authorized
a short local QLoRA run. No paid services, cloud dataset upload or public publishing.

The dedicated `.venv-soup` isolates Soup's Transformers 5 dependencies from the
earlier baseline scaffold. All model caches, source-derived training data, checkpoints
and detailed logs stay under ignored `.private/`.

Configuration is verified against the installed Soup release before running. A small
template/weak-target dataset tests training mechanics; it is not human gold and cannot
establish product quality. Always inspect the private dataset manifest and run report.

## Reproduce

Use Python 3.12 in a dedicated environment. Install CUDA PyTorch from the official
CUDA 12.8 index first, then the pinned lockfile (which retains that CUDA version).

```powershell
python -m venv .venv-soup
.\.venv-soup\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv-soup\Scripts\python.exe -m pip install -r experiments\soup\requirements-lock.txt
.\.venv-soup\Scripts\python.exe -X utf8 scripts\prepare_soup_data.py
.\.venv-soup\Scripts\python.exe -X utf8 scripts\download_soup_model.py
.\scripts\run-soup.ps1 -Stage preflight
.\scripts\run-soup.ps1 -Stage train
.\scripts\run-soup.ps1 -Stage reload
```

Dataset preparation expects the original private handout ZIP in Downloads; it does
not copy it into the public project. Model download expects the recorded author
metadata at `.private/soup-setup/qwen3-8b-metadata.json`. The exact downloaded revision
and all hashes are in the private model's `download-manifest.json`. Do not fabricate
that manifest to bypass the complete-download gate.

This is intentionally 64 optimizer steps: 64 rows, one epoch, batch1, accumulation1,
packing disabled, no automatic training validation split. The 32-example holdout
is separate. Four fixed holdout examples provide diagnostic teacher-forced losses
and prediction comparisons; they are not a gold benchmark or a generation test.

Training calls the installed Soup `SFTTrainerWrapper`, with read-only diagnostic
instrumentation for raw step losses/gradients and adapter-enabled/disabled probes.
The saved adapter is loaded into a fresh streamed model for tensor and prediction
comparison, avoiding an unsupported assumption that resident8B fits in6GB.
No Soup library files are patched. All inference-stage model loads are offline.

The configuration currently selects NVMe disk streaming because only a few GB of
system RAM were free. Do not mislabel a SATA/HDD disk as NVMe to bypass a refusal.
On another machine, use supported hardware or free memory and verify the setup.

Reports: `.private/soup-setup/preflight.json`,
`.private/soup-runs/qwen3-8b-smoke/experiment.json`, and
`.private/soup-runs/qwen3-8b-smoke/reload-verification.json`.
Existing completed adapters are preserved; choose a new output for a new experiment.
