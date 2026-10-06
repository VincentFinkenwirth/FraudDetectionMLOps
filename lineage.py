"""Which code and environment produced a model (saved in every bundle).

A model is only reproducible if I know the code, the data and the
environment. The data hashes are written by the retraining snapshot, the
library versions by train.write_bundle, and this file adds the code version.

In Jenkins the jobs run in a git checkout of the release repository, so
`git rev-parse HEAD` is the released commit. If git is not available, I use
the GIT_COMMIT variable that Jenkins sets for pipelines from SCM.
"""

import os
import platform
import subprocess
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


def run_git(*args):
    """Output of a git command in the project folder, or None if that does not work."""
    try:
        result = subprocess.run(["git", *args], cwd=PROJECT_DIR, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def code_lineage():
    commit = run_git("rev-parse", "HEAD")
    if commit is not None:
        # Changes to tracked files that are not committed (only happens locally;
        # Jenkins always runs a clean checkout of a released commit).
        dirty = bool(run_git("status", "--porcelain", "--untracked-files=no"))
    else:
        commit, dirty = os.environ.get("GIT_COMMIT"), None
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        # BUILD_TAG is set by Jenkins, e.g. "jenkins-fraud-retrain-12".
        "trained_by": os.environ.get("BUILD_TAG", "local"),
        "python_version": platform.python_version(),
    }
