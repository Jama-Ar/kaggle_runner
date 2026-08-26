# Kaggle Runner

CLI-based execution layer for running arbitrary project commands on Kaggle CPU notebooks.

## Features

- Submit 1 to 5 jobs in parallel
- YAML-based job definitions
- Submit selected jobs by ID
- Generic multi-line shell commands
- Automatic project source synchronization
- Worker status monitoring
- Persistent Kaggle outputs
- Automatic result archival before worker reuse
- Manual result retrieval by job ID
- Windows and WSL compatible

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Authenticate with Kaggle:

```bash
kaggle auth login
```

Create a local job file from `jobs.example.yaml`.

## Usage

Submit all jobs:

```bash
python submit_kaggle.py jobs.yaml
```

Submit one job:

```bash
python submit_kaggle.py jobs.yaml --id JOB_ID
```

Show submitted commands:

```bash
python submit_kaggle.py jobs.yaml --verbose
```

Check workers:

```bash
python submit_kaggle.py --status
```

Extended status:

```bash
python submit_kaggle.py --status --verbose
```

Download results:

```bash
python submit_kaggle.py --results JOB_ID
```

## Configuration

`runner.yaml` defines the project repository, Git branch, Kaggle source dataset, and worker pool.

Commands defined in the job YAML are passed unchanged to the Kaggle execution environment and run under Linux.

## Concurrency

The configured worker pool supports up to five concurrent jobs.

Do not submit concurrently from multiple machines at exactly the same time. Sequential use across different machines is supported.

## Results

Kaggle remains the persistent remote source of run outputs.

Downloaded results are archived locally under:

```text
results/
```

Each execution receives a unique execution ID, so repeated submissions of the same job ID do not overwrite each other.
