# Kaggle Runner

Generic CLI-based execution layer for running project jobs on Kaggle notebooks.

## Workflow

The repository contains only generic runner logic and templates. Project-specific settings live in the untracked `runner.yaml`; job definitions live in the untracked `jobs.yaml`.

A fresh setup is:

```bash
pip install -r requirements.txt
kaggle auth login
cp runner.example.yaml runner.yaml
cp jobs.example.yaml jobs.yaml
```

PowerShell:

```powershell
Copy-Item runner.example.yaml runner.yaml
Copy-Item jobs.example.yaml jobs.yaml
```

Edit `runner.yaml` for the project and Kaggle workspace, then initialize the remote resources once:

```bash
python submit_kaggle.py --init
```

`--init` is idempotent. Missing resources are created; existing source datasets and workers are detected and left unchanged. Re-running it on an initialized workspace only reports the existing resources.

## Project setup

`project.setup_command` is optional. Use `null` when no installation or setup is needed:

```yaml
project:
  setup_command: null
```

Or provide any Linux shell setup command:

```yaml
project:
  setup_command: |
    python -m pip install -e .
```

Each source snapshot contains a runner-owned `kaggle_runner_source.json` marker at the project root. The Kaggle notebook resolves the project root from this marker, so source discovery does not depend on project-specific files or on whether Kaggle exposes an uploaded ZIP as an archive or as expanded files.

The setup command runs inside `project.kaggle_workdir` before each job command. Setup uses fail-fast shell semantics, so any failing setup command aborts the notebook instead of being masked by a later successful command.

## Templates

Tracked generic templates:

- `templates/runner.ipynb`
- `templates/kernel-metadata.json`
- `templates/dataset-metadata.json`

The notebook template requires these marker cells:

- `# Runner Configuration`
- `# Job Definition`
- `# Job Execution`

Additional notebook cells may be added around them. Before submission, the runner validates and restores both the standard Python 3 `kernelspec` and Kaggle's notebook metadata (`metadata.kaggle.language=python`, `metadata.kaggle.sourceType=notebook`).

## Jobs

Example:

```yaml
jobs:
  - id: example_job
    command: |
      echo "Hello from Kaggle"
```

Submit all jobs:

```bash
python submit_kaggle.py jobs.yaml
```

Submit selected jobs:

```bash
python submit_kaggle.py jobs.yaml --id JOB_ID
```

## Monitoring and results

```bash
python submit_kaggle.py --status
python submit_kaggle.py --status --verbose
python submit_kaggle.py --results JOB_ID
```

Kaggle stores notebook outputs remotely. Downloaded outputs are archived locally under `results/`.

New archives use the naming scheme:

```text
<job-id>__worker-<n>__v<kaggle-version>__<execution-id>.zip
```

Example:

```text
test_01__worker-1__v11__20260827T103751Z_bdcf68ee.zip
```

The execution ID contains a UTC timestamp plus a random suffix and remains the authoritative uniqueness component. The Kaggle version is included for readability. If the version is unavailable on the current machine, `vunknown` is used while the execution ID still keeps the archive unique.

Existing archives keep their original filenames and are recognized by their embedded execution ID, so applying a new naming scheme does not duplicate already archived runs.

## Local project files

The following are intentionally not tracked:

- `runner.yaml`
- `jobs.yaml`
- `.kaggle/`
- `results/`
- `submission_history.jsonl`

This keeps the repository independent of any specific application.
