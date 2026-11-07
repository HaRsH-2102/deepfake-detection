# Project Report Builder

This folder contains a reusable report generator for the deepfake detection project.

## What it produces

Running the builder creates a self-contained report package in `project_report/generated/`:

- `deepfake_detection_project_report.md`
- `deepfake_detection_project_report.html`
- `assets/` with copied figures
- `artifacts/` with copied JSON summaries
- `manifest.json`

## Default command

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m project_report.build_report
```

## Custom paths

```powershell
.\.venv\Scripts\python.exe -m project_report.build_report `
  --run-dir "own_train\runs\20260422_231937_xception" `
  --component-dir "outputs\component_checks" `
  --output-dir "project_report\generated"
```

## Export guidance

- Open `project_report/generated/deepfake_detection_project_report.html` in a browser and use Print to PDF for a polished printable version.
- Use the Markdown file if you want to paste the report into Word, Google Docs, Overleaf, or another submission workflow.
- The copied `assets/` and `artifacts/` folders make the report portable across systems.
