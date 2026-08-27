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

The setup command runs inside `project.kaggle_workdir` before each job command.

## Templates

Tracked generic templates:

- `templates/runner.ipynb`
- `templates/kernel-metadata.json`
- `templates/dataset-metadata.json`

The notebook template requires these marker cells:

- `# Runner Configuration`
- `# Job Definition`
- `# Job Execution`

Additional notebook cells may be added around them.

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

## Local project files

The following are intentionally not tracked:

- `runner.yaml`
- `jobs.yaml`
- `.kaggle/`
- `results/`
- `submission_history.jsonl`

This keeps the repository independent of any specific application.
