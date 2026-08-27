import argparse
import hashlib
import json
import os
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

SOURCE_FORMAT_VERSION = 2
SOURCE_MARKER_NAME = "kaggle_runner_source.json"

DATASET = f"{USERNAME}/{KAGGLE_CONFIG['source_dataset']}"

WORKER_PREFIX = KAGGLE_CONFIG["worker_prefix"]
WORKER_COUNT = int(KAGGLE_CONFIG["worker_count"])

NOTEBOOK_TEMPLATE = (
    ROOT / KAGGLE_CONFIG["notebook_template"]
).resolve()

KERNEL_METADATA_TEMPLATE = (
    ROOT / KAGGLE_CONFIG["kernel_metadata_template"]
).resolve()

DATASET_METADATA_TEMPLATE = (
    ROOT / KAGGLE_CONFIG["dataset_metadata_template"]
).resolve()

PROJECT_WORKDIR = PROJECT_CONFIG.get(
    "kaggle_workdir",
    "/kaggle/temp/project",
)

SETUP_COMMAND = PROJECT_CONFIG.get("setup_command")

if SETUP_COMMAND is not None:
    if not isinstance(SETUP_COMMAND, str):
        raise ValueError(
            "project.setup_command must be a string or null."
        )

    if not SETUP_COMMAND.strip():
        SETUP_COMMAND = None

WORKERS = [
    {
        "number": i,
        "kernel": f"{USERNAME}/{WORKER_PREFIX}-{i}",
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



def get_remote_source_manifest():
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

        manifests = list(
            temp_dir.rglob("source_manifest.json")
        )

        if not manifests:
            return None

        try:
            data = json.loads(
                manifests[0].read_text(
                    encoding="utf-8"
                )
            )
        except (json.JSONDecodeError, OSError):
            return None

        return data if isinstance(data, dict) else None




def write_dataset_metadata():
    try:
        metadata = json.loads(
            DATASET_METADATA_TEMPLATE.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(
            f"Could not read dataset metadata template: "
            f"{DATASET_METADATA_TEMPLATE}"
        ) from exc

    metadata["id"] = DATASET
    metadata["title"] = KAGGLE_CONFIG.get(
        "source_title",
        KAGGLE_CONFIG["source_dataset"],
    )

    (SOURCE_PATH / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

def add_source_marker_to_archive(commit):
    marker = {
        "format_version": SOURCE_FORMAT_VERSION,
        "commit": commit,
        "remote": REMOTE,
        "branch": BRANCH,
    }

    with zipfile.ZipFile(
        SOURCE_ZIP_PATH,
        mode="a",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr(
            SOURCE_MARKER_NAME,
            json.dumps(
                marker,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
        )

    with zipfile.ZipFile(
        SOURCE_ZIP_PATH,
        mode="r",
    ) as archive:
        if SOURCE_MARKER_NAME not in archive.namelist():
            raise RuntimeError(
                "Source marker was not written to source archive."
            )

        try:
            stored_marker = json.loads(
                archive.read(
                    SOURCE_MARKER_NAME
                ).decode("utf-8")
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise RuntimeError(
                "Source marker in archive is invalid."
            ) from exc

    if stored_marker.get("commit") != commit:
        raise RuntimeError(
            "Source marker commit does not match archive commit."
        )

def update_source_if_needed(commit, worker_states):
    remote_manifest = get_remote_source_manifest()

    remote_commit = (
        remote_manifest.get("commit")
        if isinstance(remote_manifest, dict)
        else None
    )

    remote_format_version = (
        remote_manifest.get("format_version")
        if isinstance(remote_manifest, dict)
        else None
    )

    if (
        remote_commit == commit
        and remote_format_version == SOURCE_FORMAT_VERSION
    ):
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

    SOURCE_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_dataset_metadata()

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

    add_source_marker_to_archive(commit)

    SOURCE_MANIFEST_PATH.write_text(
        json.dumps(
            {
                "format_version": SOURCE_FORMAT_VERSION,
                "commit": commit,
                "remote": REMOTE,
                "branch": BRANCH,
                "marker": SOURCE_MARKER_NAME,
                "created_at": utc_now(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Source commit: {commit}")
    print(
        f"Source format: v{SOURCE_FORMAT_VERSION} "
        f"({SOURCE_MARKER_NAME})"
    )
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






def ensure_notebook_kernel_metadata(
    notebook,
    kernel_metadata,
):
    kernelspec = notebook.metadata.get("kernelspec")

    if not isinstance(kernelspec, dict):
        kernelspec = {}

    kernelspec["display_name"] = "Python 3"
    kernelspec["language"] = "python"
    kernelspec["name"] = "python3"

    notebook.metadata["kernelspec"] = kernelspec

    language_info = notebook.metadata.get(
        "language_info"
    )

    if not isinstance(language_info, dict):
        language_info = {}

    language_info.update(
        {
            "name": "python",
            "mimetype": "text/x-python",
            "file_extension": ".py",
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "codemirror_mode": {
                "name": "ipython",
                "version": 3,
            },
        }
    )

    notebook.metadata["language_info"] = language_info

    kaggle_metadata = notebook.metadata.get("kaggle")

    if not isinstance(kaggle_metadata, dict):
        kaggle_metadata = {}

    enable_gpu = bool(
        kernel_metadata.get("enable_gpu", False)
    )

    enable_internet = bool(
        kernel_metadata.get("enable_internet", False)
    )

    kaggle_metadata.update(
        {
            "accelerator": (
                "gpu"
                if enable_gpu
                else "none"
            ),
            "dataSources": [],
            "isGpuEnabled": enable_gpu,
            "isInternetEnabled": enable_internet,
            "language": "python",
            "sourceType": "notebook",
        }
    )

    notebook.metadata["kaggle"] = kaggle_metadata

    nbformat.validate(notebook)

    if (
        notebook.metadata["kernelspec"].get("name")
        != "python3"
    ):
        raise RuntimeError(
            "Generated notebook has no valid Python "
            "kernelspec."
        )

    if (
        notebook.metadata["kaggle"].get("language")
        != "python"
        or notebook.metadata["kaggle"].get("sourceType")
        != "notebook"
    ):
        raise RuntimeError(
            "Generated notebook has invalid Kaggle "
            "notebook metadata."
        )

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

    try:
        kernel_metadata = json.loads(
            KERNEL_METADATA_TEMPLATE.read_text(
                encoding="utf-8"
            )
        )
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(
            f"Could not read kernel metadata template: "
            f"{KERNEL_METADATA_TEMPLATE}"
        ) from exc

    code_file = kernel_metadata.get(
        "code_file",
        "runner.ipynb",
    )

    if (
        not isinstance(code_file, str)
        or not code_file
        or Path(code_file).name != code_file
    ):
        raise ValueError(
            "kernel metadata template requires a simple "
            "'code_file' filename."
        )

    notebook_target = temp_dir / code_file
    metadata_target = temp_dir / "kernel-metadata.json"

    shutil.copy2(
        NOTEBOOK_TEMPLATE,
        notebook_target,
    )

    kernel_metadata["id"] = worker["kernel"]
    kernel_metadata["title"] = (
        f"{WORKER_PREFIX}-{worker['number']}"
    )

    dataset_sources = kernel_metadata.get(
        "dataset_sources",
        [],
    )

    if not isinstance(dataset_sources, list):
        raise ValueError(
            "kernel metadata template 'dataset_sources' "
            "must be a list."
        )

    kernel_metadata["dataset_sources"] = list(
        dict.fromkeys(
            [DATASET, *dataset_sources]
        )
    )

    metadata_target.write_text(
        json.dumps(
            kernel_metadata,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    notebook = nbformat.read(
        notebook_target,
        as_version=4,
    )

    ensure_notebook_kernel_metadata(
        notebook,
        kernel_metadata,
    )

    configuration_cell = None
    definition_cell = None
    execution_cell = None

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue

        if "# Runner Configuration" in cell.source:
            configuration_cell = cell

        if "# Job Definition" in cell.source:
            definition_cell = cell

        if "# Job Execution" in cell.source:
            execution_cell = cell

    if configuration_cell is None:
        raise RuntimeError(
            "Could not find '# Runner Configuration' cell."
        )

    if definition_cell is None:
        raise RuntimeError(
            "Could not find '# Job Definition' cell."
        )

    if execution_cell is None:
        raise RuntimeError(
            "Could not find '# Job Execution' cell."
        )

    configuration_cell.source = (
        "# Runner Configuration\n\n"
        f"PROJECT_WORKDIR = {PROJECT_WORKDIR!r}\n"
        f"SETUP_COMMAND = {SETUP_COMMAND!r}\n"
        f"SOURCE_COMMIT = {commit!r}"
    )

    definition_cell.source = (
        "# Job Definition\n\n"
        f"JOB_ID = {job['id']!r}\n"
        f"EXECUTION_ID = {execution_id!r}\n"
        f"SUBMITTED_AT = {submitted_at!r}\n"
        f"WORKER_NUMBER = {worker['number']!r}\n"
        f"COMMAND = {job['command']!r}"
    )

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




def get_kaggle_version_for_metadata(worker, metadata):
    execution_id = metadata.get("execution_id")

    if not isinstance(execution_id, str) or not execution_id:
        return None

    history = load_latest_history_by_worker()
    record = history.get(worker["number"])

    if not isinstance(record, dict):
        return None

    if record.get("execution_id") != execution_id:
        return None

    version = record.get("version")

    if isinstance(version, bool):
        return None

    try:
        version = int(version)
    except (TypeError, ValueError):
        return None

    return version if version > 0 else None


def find_local_archive_for_execution(metadata):
    execution_id = metadata.get("execution_id")

    if (
        not isinstance(execution_id, str)
        or not execution_id
        or not RESULTS_PATH.exists()
    ):
        return None

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

                archived_metadata = json.loads(
                    archive.read(target_name)
                )

                if (
                    archived_metadata.get("execution_id")
                    == execution_id
                ):
                    return archive_path
        except (
            zipfile.BadZipFile,
            json.JSONDecodeError,
            OSError,
        ):
            continue

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

    version = get_kaggle_version_for_metadata(
        worker,
        metadata,
    )

    version_label = (
        f"v{version}"
        if version is not None
        else "vunknown"
    )

    return (
        RESULTS_PATH
        / (
            f"{job_id}"
            f"__worker-{worker['number']}"
            f"__{version_label}"
            f"__{safe_name(execution_id)}.zip"
        )
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

    existing_archive = find_local_archive_for_execution(
        metadata
    )

    if existing_archive is not None:
        if verbose:
            print(
                f"Result already archived: {existing_archive}"
            )

        return existing_archive, metadata

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
            "kaggle_version": get_kaggle_version_for_metadata(
                worker,
                metadata_after,
            ),
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



def _run_kaggle_process(command, timeout=120):
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        return subprocess.run(
            command,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Timed out running: {' '.join(map(str, command))}"
        ) from exc


def _kaggle_error_text(result):
    return (
        result.stderr.strip()
        or result.stdout.strip()
        or f"Kaggle CLI exit code {result.returncode}."
    )


def _probe_kaggle_resource(command, resource_name):
    result = _run_kaggle_process(command)

    if result.returncode == 0:
        return True

    details = _kaggle_error_text(result)
    normalized = details.lower()

    not_found_markers = (
        "404",
        "not found",
        "does not exist",
    )

    if any(marker in normalized for marker in not_found_markers):
        return False

    raise RuntimeError(
        f"Could not check {resource_name}: {details}"
    )


def source_dataset_exists():
    return _probe_kaggle_resource(
        [
            "kaggle",
            "datasets",
            "status",
            DATASET,
        ],
        f"source dataset '{DATASET}'",
    )


def worker_exists(worker):
    return _probe_kaggle_resource(
        [
            "kaggle",
            "kernels",
            "status",
            worker["kernel"],
        ],
        f"worker '{worker['kernel']}'",
    )


def write_initial_source_snapshot(commit):
    SOURCE_PATH.mkdir(parents=True, exist_ok=True)
    write_dataset_metadata()

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


def wait_for_source_dataset(timeout=300):
    deadline = time.time() + timeout

    while True:
        result = _run_kaggle_process(
            [
                "kaggle",
                "datasets",
                "status",
                DATASET,
            ]
        )

        if result.returncode == 0:
            status = result.stdout.strip().lower()

            if status == "ready":
                return

        if time.time() >= deadline:
            raise RuntimeError(
                f"Timed out waiting for source dataset '{DATASET}'."
            )

        time.sleep(3)


def create_source_dataset(commit):
    write_initial_source_snapshot(commit)

    print(f"Creating private source dataset: {DATASET}")

    result = _run_kaggle_process(
        [
            "kaggle",
            "datasets",
            "create",
            "-p",
            str(SOURCE_PATH),
            "-q",
        ],
        timeout=300,
    )

    if result.returncode != 0:
        # Kaggle CLI can fail after a successful remote action on some
        # Windows console encodings. Verify remote state before failing.
        if not source_dataset_exists():
            raise RuntimeError(
                f"Could not create source dataset '{DATASET}': "
                f"{_kaggle_error_text(result)}"
            )

    wait_for_source_dataset()
    print(f"Source dataset ready: {DATASET}")


def create_worker_initialization_directory(worker):
    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=f"kaggle_init_worker_{worker['number']}_"
        )
    )

    try:
        kernel_metadata = json.loads(
            KERNEL_METADATA_TEMPLATE.read_text(
                encoding="utf-8"
            )
        )
    except (json.JSONDecodeError, OSError) as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(
            f"Could not read kernel metadata template: "
            f"{KERNEL_METADATA_TEMPLATE}"
        ) from exc

    code_file = kernel_metadata.get(
        "code_file",
        "runner.ipynb",
    )

    if (
        not isinstance(code_file, str)
        or not code_file
        or Path(code_file).name != code_file
    ):
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise ValueError(
            "kernel metadata template requires a simple "
            "'code_file' filename."
        )

    kernel_metadata["id"] = worker["kernel"]
    kernel_metadata["title"] = (
        f"{WORKER_PREFIX}-{worker['number']}"
    )

    dataset_sources = kernel_metadata.get(
        "dataset_sources",
        [],
    )

    if not isinstance(dataset_sources, list):
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise ValueError(
            "kernel metadata template 'dataset_sources' "
            "must be a list."
        )

    kernel_metadata["dataset_sources"] = list(
        dict.fromkeys(
            [DATASET, *dataset_sources]
        )
    )

    (temp_dir / "kernel-metadata.json").write_text(
        json.dumps(
            kernel_metadata,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_code_cell(
            "# Worker Initialization\n\n"
            "print('Kaggle runner worker initialized.')"
        )
    ]

    nbformat.write(
        notebook,
        temp_dir / code_file,
    )

    return temp_dir


def create_worker(worker):
    temp_dir = create_worker_initialization_directory(worker)

    try:
        print(
            f"Creating worker-{worker['number']}: "
            f"{worker['kernel']}"
        )

        result = _run_kaggle_process(
            [
                "kaggle",
                "kernels",
                "push",
                "-p",
                str(temp_dir),
            ],
            timeout=300,
        )

        if result.returncode != 0:
            if not worker_exists(worker):
                raise RuntimeError(
                    f"Could not create worker-{worker['number']}: "
                    f"{_kaggle_error_text(result)}"
                )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def wait_for_worker_initialization(workers, timeout=600):
    if not workers:
        return

    pending = {
        worker["number"]: worker
        for worker in workers
    }
    deadline = time.time() + timeout

    while pending:
        for number, worker in list(pending.items()):
            status = get_worker_status(worker)

            if status == "COMPLETE":
                print(f"worker-{number}: ready")
                del pending[number]
                continue

            if status in {"ERROR", "CANCELLED"}:
                raise RuntimeError(
                    f"worker-{number} initialization ended with "
                    f"status {status}. Check its Kaggle logs."
                )

        if not pending:
            return

        if time.time() >= deadline:
            pending_text = ", ".join(
                f"worker-{number}"
                for number in pending
            )
            raise RuntimeError(
                "Timed out waiting for worker initialization: "
                f"{pending_text}"
            )

        time.sleep(3)


def validate_initialization_files():
    required_paths = [
        REPO_PATH,
        NOTEBOOK_TEMPLATE,
        KERNEL_METADATA_TEMPLATE,
        DATASET_METADATA_TEMPLATE,
    ]

    missing = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing:
        missing_text = ", ".join(
            str(path)
            for path in missing
        )
        raise RuntimeError(
            f"Initialization requires existing paths: {missing_text}"
        )


def initialize_workspace(verbose=False):
    validate_initialization_files()

    print("Checking Kaggle workspace...")

    dataset_present = source_dataset_exists()
    worker_presence = {
        worker["number"]: worker_exists(worker)
        for worker in WORKERS
    }

    if dataset_present:
        print(f"Source dataset: existing ({DATASET})")
    else:
        print(f"Source dataset: missing ({DATASET})")

    for worker in WORKERS:
        state = "existing" if worker_presence[worker["number"]] else "missing"
        print(
            f"worker-{worker['number']}: {state} "
            f"({worker['kernel']})"
        )

    all_workers_present = all(worker_presence.values())

    if dataset_present and all_workers_present:
        print()
        print("Workspace already initialized. No changes made.")
        print(f"Source dataset: {DATASET}")
        print(f"Workers: {WORKER_COUNT}/{WORKER_COUNT} existing")
        return

    if not dataset_present:
        commit = get_latest_project_commit()
        create_source_dataset(commit)

    created_workers = []

    for worker in WORKERS:
        if worker_presence[worker["number"]]:
            continue

        create_worker(worker)
        created_workers.append(worker)

    wait_for_worker_initialization(created_workers)

    print()
    print("Workspace initialization complete.")
    print(f"Source dataset: {DATASET}")
    print(f"Workers: {WORKER_COUNT}/{WORKER_COUNT} existing")

    if verbose and created_workers:
        created_text = ", ".join(
            f"worker-{worker['number']}"
            for worker in created_workers
        )
        print(f"Created workers: {created_text}")



def main():
    parser = argparse.ArgumentParser(
        description=(
            "Initialize, submit, monitor, and collect Kaggle jobs."
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
        "--init",
        action="store_true",
        help=(
            "Initialize missing Kaggle source and worker "
            "resources without modifying existing ones."
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

    if args.init:
        if args.job_file:
            parser.error(
                "Do not provide a job file with --init."
            )

        if args.job_ids:
            parser.error(
                "--id cannot be combined with --init."
            )

        if args.status:
            parser.error(
                "--status cannot be combined with --init."
            )

        if args.results:
            parser.error(
                "--results cannot be combined with --init."
            )

        initialize_workspace(
            verbose=args.verbose,
        )

        return

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
