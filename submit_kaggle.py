import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import nbformat
import yaml


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "runner.yaml"
HISTORY_PATH = ROOT / "submission_history.jsonl"
RESULTS_PATH = ROOT / "results"


ACTIVE_STATUSES = {
    "RUNNING",
    "QUEUED",
    "PENDING",
}

FREE_STATUSES = {
    "COMPLETE",
    "ERROR",
    "CANCELLED",
}


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_execution_id():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{uuid.uuid4().hex[:8]}"


def safe_name(value):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "job"


def load_runner_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("runner.yaml is invalid.")

    if "project" not in config or "kaggle" not in config:
        raise ValueError(
            "runner.yaml requires 'project' and 'kaggle' sections."
        )

    return config


CONFIG = load_runner_config()

PROJECT_CONFIG = CONFIG["project"]
KAGGLE_CONFIG = CONFIG["kaggle"]

REPO_PATH = (ROOT / PROJECT_CONFIG["repo_path"]).resolve()
REMOTE = PROJECT_CONFIG.get("remote", "origin")
BRANCH = PROJECT_CONFIG["branch"]

USERNAME = KAGGLE_CONFIG["username"]

SOURCE_PATH = (ROOT / KAGGLE_CONFIG["source_dir"]).resolve()
SOURCE_ZIP_PATH = SOURCE_PATH / "source.zip"
SOURCE_MANIFEST_PATH = SOURCE_PATH / "source_manifest.json"

DATASET = f"{USERNAME}/{KAGGLE_CONFIG['source_dataset']}"

WORKER_PREFIX = KAGGLE_CONFIG["worker_prefix"]
WORKER_COUNT = int(KAGGLE_CONFIG["worker_count"])

WORKERS = [
    {
        "number": i,
        "kernel": f"{USERNAME}/{WORKER_PREFIX}-{i}",
        "directory": ROOT / f"worker_{i}",
    }
    for i in range(1, WORKER_COUNT + 1)
]


def run(command, cwd=None, timeout=None):
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=True,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as exc:
        details = exc.stderr.strip() or exc.stdout.strip()
        raise RuntimeError(
            f"Command failed: {' '.join(map(str, command))}\n{details}"
        ) from exc

    return result.stdout.strip()


def run_optional(command, timeout=15):
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ""

    if result.returncode != 0:
        return ""

    return result.stdout


def load_jobs(path):
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict) or "jobs" not in data:
        raise ValueError("Job file must contain a 'jobs' section.")

    jobs = data["jobs"]

    if not isinstance(jobs, list):
        raise ValueError("'jobs' must be a list.")

    if not 1 <= len(jobs) <= WORKER_COUNT:
        raise ValueError(
            f"Job file must contain between 1 and {WORKER_COUNT} jobs."
        )

    seen_ids = set()

    for index, job in enumerate(jobs, start=1):
        if not isinstance(job, dict):
            raise ValueError(f"Job {index} is invalid.")

        job_id = job.get("id")
        command = job.get("command")

        if not isinstance(job_id, str) or not job_id:
            raise ValueError(
                f"Job {index} requires a non-empty 'id'."
            )

        if job_id in seen_ids:
            raise ValueError(f"Duplicate job id: {job_id}")

        if not isinstance(command, str) or not command.strip():
            raise ValueError(
                f"Job '{job_id}' requires a non-empty 'command'."
            )

        seen_ids.add(job_id)

    return jobs


def select_jobs(jobs, requested_ids):
    if not requested_ids:
        return jobs

    jobs_by_id = {job["id"]: job for job in jobs}

    missing = [
        job_id
        for job_id in requested_ids
        if job_id not in jobs_by_id
    ]

    if missing:
        available = ", ".join(jobs_by_id)
        raise ValueError(
            f"Unknown job id(s): {', '.join(missing)}. "
            f"Available ids: {available}"
        )

    selected_ids = list(dict.fromkeys(requested_ids))

    return [jobs_by_id[job_id] for job_id in selected_ids]


def get_worker_status(worker):
    output = run(
        [
            "kaggle",
            "kernels",
            "status",
            worker["kernel"],
        ]
    )

    match = re.search(
        r"KernelWorkerStatus\.([A-Z_]+)",
        output,
    )

    if not match:
        raise RuntimeError(
            f"Could not determine status of "
            f"{worker['kernel']}: {output}"
        )

    return match.group(1)


def get_worker_states():
    return [
        {
            "worker": worker,
            "status": get_worker_status(worker),
        }
        for worker in WORKERS
    ]


def get_latest_project_commit():
    print(f"Fetching latest {REMOTE}/{BRANCH}...")

    run(
        [
            "git",
            "fetch",
            REMOTE,
            BRANCH,
        ],
        cwd=REPO_PATH,
    )

    return run(
        [
            "git",
            "rev-parse",
            f"{REMOTE}/{BRANCH}",
        ],
        cwd=REPO_PATH,
    )


def get_remote_source_commit():
    with tempfile.TemporaryDirectory(
        prefix="kaggle_source_manifest_"
    ) as temp_name:
        temp_dir = Path(temp_name)

        result = subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                DATASET,
                "-f",
                "source_manifest.json",
                "-p",
                str(temp_dir),
                "-o",
                "-q",
            ],
            text=True,
            capture_output=True,
            check=False,
        )

        if result.returncode != 0:
            return None

        manifests = list(temp_dir.rglob("source_manifest.json"))

        if not manifests:
            return None

        try:
            data = json.loads(
                manifests[0].read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            return None

        commit = data.get("commit")

        return commit if isinstance(commit, str) else None


def update_source_if_needed(commit, worker_states):
    remote_commit = get_remote_source_commit()

    if remote_commit == commit:
        print("Kaggle source is already up to date.")
        return

    active_workers = [
        state
        for state in worker_states
        if state["status"] in ACTIVE_STATUSES
    ]

    if active_workers:
        active_text = ", ".join(
            f"worker-{state['worker']['number']}"
            for state in active_workers
        )

        raise RuntimeError(
            "The Kaggle source version must change, but jobs are still "
            f"active on {active_text}. Wait for them to finish before "
            "switching the shared source version."
        )

    SOURCE_PATH.mkdir(parents=True, exist_ok=True)

    if SOURCE_ZIP_PATH.exists():
        SOURCE_ZIP_PATH.unlink()

    run(
        [
            "git",
            "archive",
            "--format=zip",
            f"--output={SOURCE_ZIP_PATH}",
            f"{REMOTE}/{BRANCH}",
        ],
        cwd=REPO_PATH,
    )

    SOURCE_MANIFEST_PATH.write_text(
        json.dumps(
            {
                "commit": commit,
                "remote": REMOTE,
                "branch": BRANCH,
                "created_at": utc_now(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Source commit: {commit}")
    print("Uploading source to Kaggle...")

    run(
        [
            "kaggle",
            "datasets",
            "version",
            "-p",
            str(SOURCE_PATH),
            "-m",
            f"Source commit {commit}",
        ]
    )

    deadline = time.time() + 300

    while True:
        status = run(
            [
                "kaggle",
                "datasets",
                "status",
                DATASET,
            ]
        ).strip().lower()

        if status == "ready":
            break

        if time.time() >= deadline:
            raise RuntimeError(
                "Timed out waiting for Kaggle source dataset."
            )

        print(f"Source dataset status: {status}")
        time.sleep(3)

    print("Source ready.")


def create_job_directory(
    worker,
    job,
    commit,
    execution_id,
    submitted_at,
):
    safe_job_id = safe_name(job["id"])

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=f"kaggle_{safe_job_id}_"
        )
    )

    template_notebook = (
        worker["directory"]
        / "mnist-rl-runner.ipynb"
    )

    template_metadata = (
        worker["directory"]
        / "kernel-metadata.json"
    )

    notebook_target = (
        temp_dir
        / "mnist-rl-runner.ipynb"
    )

    metadata_target = (
        temp_dir
        / "kernel-metadata.json"
    )

    shutil.copy2(template_notebook, notebook_target)
    shutil.copy2(template_metadata, metadata_target)

    notebook = nbformat.read(
        notebook_target,
        as_version=4,
    )

    definition_cell = None
    execution_cell = None

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue

        if "# Job Definition" in cell.source:
            definition_cell = cell

        if "# Job Execution" in cell.source:
            execution_cell = cell

    if definition_cell is None:
        raise RuntimeError(
            "Could not find '# Job Definition' cell."
        )

    if execution_cell is None:
        raise RuntimeError(
            "Could not find '# Job Execution' cell."
        )

    definition_cell.source = (
        "# Job Definition\n\n"
        f"JOB_ID = {job['id']!r}\n"
        f"EXECUTION_ID = {execution_id!r}\n"
        f"SUBMITTED_AT = {submitted_at!r}\n"
        f"WORKER_NUMBER = {worker['number']!r}\n"
        f"SOURCE_COMMIT = {commit!r}\n"
        f"COMMAND = {job['command']!r}"
    )

    execution_cell.source = """# Job Execution

import json
import subprocess
from pathlib import Path

job_metadata = {
    "job_id": JOB_ID,
    "execution_id": EXECUTION_ID,
    "submitted_at": SUBMITTED_AT,
    "worker": WORKER_NUMBER,
    "source_commit": SOURCE_COMMIT,
    "command": COMMAND,
}

Path("/kaggle/working/job_metadata.json").write_text(
    json.dumps(job_metadata, indent=2),
    encoding="utf-8",
)

subprocess.run(
    ["bash", "-c", COMMAND],
    check=True,
    cwd="/kaggle/temp/mnist_active_screening",
)
"""

    nbformat.write(
        notebook,
        notebook_target,
    )

    return temp_dir


def save_history(record):
    with open(
        HISTORY_PATH,
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


def load_latest_history_by_worker():
    latest = {}

    if not HISTORY_PATH.exists():
        return latest

    with open(
        HISTORY_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            worker_number = record.get("worker")

            if isinstance(worker_number, int):
                latest[worker_number] = record

    return latest


def metadata_fingerprint(metadata):
    if metadata is None:
        return None

    payload = json.dumps(
        metadata,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def read_job_metadata_from_directory(directory):
    files = list(Path(directory).rglob("job_metadata.json"))

    if not files:
        return None

    target = min(
        files,
        key=lambda path: len(path.parts),
    )

    try:
        return json.loads(
            target.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError):
        return None



def get_current_job_metadata(worker):
    with tempfile.TemporaryDirectory(
        prefix=f"kaggle_meta_worker_{worker['number']}_"
    ) as temp_name:
        temp_dir = Path(temp_name)

        result = subprocess.run(
            [
                "kaggle",
                "kernels",
                "output",
                worker["kernel"],
                "-p",
                str(temp_dir),
                "-o",
                "--file-pattern",
                "job_metadata.json",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=120,
        )

        # Kaggle CLI 2.2.4 on Windows may return exit code 1
        # after the file was successfully downloaded because of
        # a console encoding error. File presence is authoritative.
        metadata = read_job_metadata_from_directory(temp_dir)

        if metadata is not None:
            return metadata

        return None



def archive_path_for_metadata(worker, metadata):
    job_id = safe_name(str(metadata.get("job_id", "job")))
    execution_id = metadata.get("execution_id")

    if not isinstance(execution_id, str) or not execution_id:
        execution_id = (
            "legacy_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "_"
            + uuid.uuid4().hex[:8]
        )

    return (
        RESULTS_PATH
        / f"{job_id}__worker-{worker['number']}__{safe_name(execution_id)}.zip"
    )



def archive_worker_output(
    worker,
    expected_job_id=None,
    verbose=False,
):
    status = get_worker_status(worker)

    if status in ACTIVE_STATUSES:
        return None, None

    metadata = get_current_job_metadata(worker)

    if metadata is None:
        return None, None

    if (
        expected_job_id is not None
        and metadata.get("job_id") != expected_job_id
    ):
        return None, metadata

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    archive_path = archive_path_for_metadata(
        worker,
        metadata,
    )

    if archive_path.exists():
        if verbose:
            print(f"Result already archived: {archive_path}")

        return archive_path, metadata

    fingerprint_before = metadata_fingerprint(metadata)

    with tempfile.TemporaryDirectory(
        prefix=f"kaggle_output_worker_{worker['number']}_"
    ) as temp_name:
        temp_dir = Path(temp_name)

        result = subprocess.run(
            [
                "kaggle",
                "kernels",
                "output",
                worker["kernel"],
                "-p",
                str(temp_dir),
                "-o",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=600,
        )

        # Do not trust the CLI exit code here.
        # Kaggle CLI 2.2.4 may return 1 after successfully
        # downloading the output because of Windows encoding.
        metadata_after = read_job_metadata_from_directory(
            temp_dir
        )

        if metadata_after is None:
            details = (
                result.stderr.strip()
                or result.stdout.strip()
                or "No diagnostic output."
            )

            raise RuntimeError(
                f"Output download from worker-{worker['number']} "
                f"did not produce job_metadata.json. "
                f"Kaggle CLI exit code: {result.returncode}. "
                f"{details}"
            )

        if (
            metadata_fingerprint(metadata_after)
            != fingerprint_before
        ):
            raise RuntimeError(
                f"Worker-{worker['number']} changed while its "
                "output was being archived. "
                "The worker will not be reused."
            )

        archive_manifest = {
            "archived_at": utc_now(),
            "worker": worker["number"],
            "kernel": worker["kernel"],
            "kaggle_status": status,
            "job_metadata": metadata_after,
        }

        (temp_dir / "archive_manifest.json").write_text(
            json.dumps(
                archive_manifest,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        base_name = archive_path.with_suffix("")

        shutil.make_archive(
            str(base_name),
            "zip",
            root_dir=temp_dir,
        )

    print(
        f"Archived result: "
        f"{metadata.get('job_id', 'unknown')} -> {archive_path}"
    )

    return archive_path, metadata



def prepare_workers(required, verbose=False):
    prepared = []

    for worker in WORKERS:
        if len(prepared) >= required:
            break

        status = get_worker_status(worker)

        if status not in FREE_STATUSES:
            continue

        try:
            _, metadata = archive_worker_output(
                worker,
                verbose=verbose,
            )
        except RuntimeError as exc:
            print(
                f"worker-{worker['number']} cannot be reused: {exc}"
            )
            continue

        prepared.append(
            {
                "worker": worker,
                "previous_fingerprint": metadata_fingerprint(
                    metadata
                ),
            }
        )

    if len(prepared) < required:
        raise RuntimeError(
            f"Need {required} reusable worker(s), but only "
            f"{len(prepared)} are safely available."
        )

    return prepared


def verify_worker_unchanged(prepared_worker):
    worker = prepared_worker["worker"]

    status = get_worker_status(worker)

    if status in ACTIVE_STATUSES:
        raise RuntimeError(
            f"worker-{worker['number']} became busy before submission."
        )

    current_metadata = get_current_job_metadata(worker)

    current_fingerprint = metadata_fingerprint(
        current_metadata
    )

    if (
        current_fingerprint
        != prepared_worker["previous_fingerprint"]
    ):
        raise RuntimeError(
            f"worker-{worker['number']} changed before submission. "
            "Refusing to overwrite its current run."
        )


def submit_job(
    prepared_worker,
    job,
    commit,
):
    worker = prepared_worker["worker"]

    execution_id = make_execution_id()
    submitted_at = utc_now()

    temp_dir = create_job_directory(
        worker,
        job,
        commit,
        execution_id,
        submitted_at,
    )

    try:
        verify_worker_unchanged(
            prepared_worker
        )

        output = run(
            [
                "kaggle",
                "kernels",
                "push",
                "-p",
                str(temp_dir),
            ]
        )
    finally:
        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )

    match = re.search(
        r"Kernel version (\d+)",
        output,
    )

    if not match:
        raise RuntimeError(
            f"Could not determine Kaggle version for "
            f"job '{job['id']}'. CLI output: {output}"
        )

    version = int(match.group(1))

    record = {
        "submitted_at": submitted_at,
        "job_id": job["id"],
        "execution_id": execution_id,
        "worker": worker["number"],
        "kernel": worker["kernel"],
        "version": version,
        "source_commit": commit,
        "command": job["command"],
    }

    save_history(record)

    return record


def parse_log_lines(raw):
    if not raw:
        return []

    text = raw

    try:
        records = json.loads(raw)

        if isinstance(records, list):
            text = "".join(
                str(record.get("data", ""))
                for record in records
                if isinstance(record, dict)
            )
    except json.JSONDecodeError:
        pass

    lines = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        if set(line) <= set("-=|+ "):
            continue

        lines.append(line)

    return lines


def get_recent_log_lines(worker, limit):
    raw = run_optional(
        [
            "kaggle",
            "kernels",
            "logs",
            worker["kernel"],
        ],
        timeout=15,
    )

    return parse_log_lines(raw)[-limit:]


def display_status(verbose=False):
    history = load_latest_history_by_worker()

    print("Kaggle worker status:")
    print()

    for worker in WORKERS:
        status = get_worker_status(worker)

        record = history.get(worker["number"])

        remote_metadata = None

        if status not in ACTIVE_STATUSES:
            remote_metadata = get_current_job_metadata(worker)

        job_id = "-"

        if remote_metadata is not None:
            job_id = str(
                remote_metadata.get("job_id", "-")
            )
        elif record is not None:
            job_id = str(
                record.get("job_id", "-")
            )

        version = (
            record.get("version", "-")
            if record is not None
            else "-"
        )

        display_status_name = {
            "RUNNING": "Running",
            "QUEUED": "Queued",
            "PENDING": "Pending",
            "COMPLETE": "Completed",
            "ERROR": "Error",
            "CANCELLED": "Cancelled",
        }.get(
            status,
            status.title(),
        )

        print(
            f"worker-{worker['number']} | "
            f"{display_status_name} | "
            f"job={job_id} | "
            f"version={version}"
        )

        if status in {"RUNNING", "ERROR"}:
            log_limit = 10 if verbose else 1

            lines = get_recent_log_lines(
                worker,
                log_limit,
            )

            if lines:
                if verbose:
                    print("  recent log:")

                    for line in lines:
                        print(f"    {line}")
                else:
                    print(
                        f"  latest activity: "
                        f"{lines[-1]}"
                    )

        print()


def find_local_archives(job_id):
    if not RESULTS_PATH.exists():
        return []

    matches = []

    for archive_path in RESULTS_PATH.glob("*.zip"):
        try:
            with zipfile.ZipFile(
                archive_path,
                "r",
            ) as archive:
                metadata_names = [
                    name
                    for name in archive.namelist()
                    if name.endswith("job_metadata.json")
                ]

                if not metadata_names:
                    continue

                target_name = min(
                    metadata_names,
                    key=lambda name: len(
                        Path(name).parts
                    ),
                )

                metadata = json.loads(
                    archive.read(target_name)
                )

                if metadata.get("job_id") == job_id:
                    matches.append(archive_path)
        except (
            zipfile.BadZipFile,
            json.JSONDecodeError,
            OSError,
        ):
            continue

    return sorted(matches)


def collect_results(job_id, verbose=False):
    existing = find_local_archives(job_id)

    found_remote = []

    for worker in WORKERS:
        status = get_worker_status(worker)

        if status in ACTIVE_STATUSES:
            continue

        metadata = get_current_job_metadata(worker)

        if metadata is None:
            continue

        if metadata.get("job_id") != job_id:
            continue

        archive_path, _ = archive_worker_output(
            worker,
            expected_job_id=job_id,
            verbose=verbose,
        )

        if archive_path is not None:
            found_remote.append(archive_path)

    combined = []

    for path in existing + found_remote:
        if path not in combined:
            combined.append(path)

    if combined:
        print(f"Results for job '{job_id}':")

        for path in combined:
            print(f"  {path}")

        return

    history = load_latest_history_by_worker()

    running_workers = []

    for worker in WORKERS:
        record = history.get(worker["number"])

        if (
            record is not None
            and record.get("job_id") == job_id
            and get_worker_status(worker) in ACTIVE_STATUSES
        ):
            running_workers.append(worker["number"])

    if running_workers:
        workers_text = ", ".join(
            f"worker-{number}"
            for number in running_workers
        )

        raise RuntimeError(
            f"Job '{job_id}' is still active on {workers_text}."
        )

    raise RuntimeError(
        f"Job '{job_id}' was not found in local archives or in the "
        "current output of any worker. If that worker has already been "
        "reused on another machine, check that machine's results archive "
        "or the historical Kaggle notebook version."
    )


def submit_batch(
    job_file,
    requested_ids,
    verbose,
):
    jobs = load_jobs(job_file)

    jobs = select_jobs(
        jobs,
        requested_ids,
    )

    print(f"Selected {len(jobs)} job(s):")

    for job in jobs:
        print(f"  - {job['id']}")

    worker_states = get_worker_states()

    commit = get_latest_project_commit()

    update_source_if_needed(
        commit,
        worker_states,
    )

    prepared_workers = prepare_workers(
        len(jobs),
        verbose=verbose,
    )

    print()
    print("Submitting jobs...")
    print()

    for job, prepared_worker in zip(
        jobs,
        prepared_workers,
    ):
        record = submit_job(
            prepared_worker,
            job,
            commit,
        )

        print(
            f"  {job['id']} -> "
            f"worker-{record['worker']} -> "
            f"version {record['version']}"
        )

        if verbose:
            print()
            print(
                job["command"],
                end=(
                    ""
                    if job["command"].endswith("\n")
                    else "\n"
                ),
            )
            print()

    print()
    print("Batch submitted successfully.")
    print(f"Source commit: {commit}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Submit, monitor, and collect Kaggle jobs."
        )
    )

    parser.add_argument(
        "job_file",
        nargs="?",
        help="Path to YAML job definition file.",
    )

    parser.add_argument(
        "--id",
        dest="job_ids",
        action="append",
        help=(
            "Submit only the specified job id. "
            "Can be used multiple times."
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Show submitted commands or extended "
            "status/result information."
        ),
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current status of all Kaggle workers.",
    )

    parser.add_argument(
        "--results",
        metavar="JOB_ID",
        help=(
            "Download and archive the current Kaggle output "
            "for the specified job id."
        ),
    )

    args = parser.parse_args()

    if args.status:
        if args.job_file:
            parser.error(
                "Do not provide a job file with --status."
            )

        if args.job_ids:
            parser.error(
                "--id cannot be combined with --status."
            )

        if args.results:
            parser.error(
                "--results cannot be combined with --status."
            )

        display_status(
            verbose=args.verbose,
        )

        return

    if args.results:
        if args.job_file:
            parser.error(
                "Do not provide a job file with --results."
            )

        if args.job_ids:
            parser.error(
                "--id cannot be combined with --results."
            )

        collect_results(
            args.results,
            verbose=args.verbose,
        )

        return

    if not args.job_file:
        parser.error(
            "A job file is required for submission."
        )

    submit_batch(
        Path(args.job_file),
        args.job_ids,
        args.verbose,
    )


if __name__ == "__main__":
    main()
