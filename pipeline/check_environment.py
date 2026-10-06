"""CI step: is the installed environment exactly the one in the lock file?

    python -m pipeline.check_environment                 # checks requirements.lock
    python -m pipeline.check_environment --lock requirements-api.lock

requirements.txt only lists the libraries I use directly, with ranges.
requirements.lock is generated from it (uv pip compile, Linux + Python 3.12)
and pins EVERY package, also the ones that are only installed as a dependency.
The Jenkins image and the API image both install from a lock file, so a model
is trained and served with the same library versions.
"""

import argparse
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def normalise(name):
    """pip treats Scikit_Learn, scikit.learn and scikit-learn as the same package."""
    return re.sub(r"[-_.]+", "-", name).lower()


def read_lock(path=None, lines=None):
    """{package: version} from the name==version lines of a lock file."""
    if lines is None:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    pins = {}
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            name, pinned = line.split("==", 1)
            # e.g. "psycopg[binary]==3.2" -> psycopg; markers after ";" are ignored
            pins[normalise(name.split("[")[0].strip())] = pinned.split(";")[0].strip()
    return pins


def installed_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def mismatches(lock, get_version=installed_version):
    """{package: (locked, installed)} for every package that is not installed exactly as locked."""
    found = {}
    for name, locked in lock.items():
        installed = get_version(name)
        if installed != locked:
            found[name] = (locked, installed)
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lock", type=Path, default=PROJECT_DIR / "requirements.lock")
    args = parser.parse_args()

    lock = read_lock(args.lock)
    found = mismatches(lock)
    for name, (locked, installed) in sorted(found.items()):
        print(f"MISMATCH {name}: locked {locked}, installed {installed}")
    if found:
        sys.exit(f"{len(found)} of {len(lock)} packages differ from {args.lock.name}.")
    print(f"All {len(lock)} packages match {args.lock.name}.")


if __name__ == "__main__":
    main()
