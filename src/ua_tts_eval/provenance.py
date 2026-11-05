"""Facts about the code and the machine that produced a result."""

import json
import os
import platform
import re
import subprocess
from importlib import metadata
from pathlib import Path

URL_CREDENTIALS = re.compile(r"(?<=://)[^/@]+@")


def _run(*command: str, cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _git_commit(folder: Path) -> str | None:
    return _run("git", "rev-parse", "--short", "HEAD", cwd=folder)


def model_source(folder: Path) -> dict:
    """Repo and commit of the model's clone. Results are published, so a token in the remote URL is dropped."""
    remote = _run("git", "remote", "get-url", "origin", cwd=folder)
    return {
        "repo": URL_CREDENTIALS.sub("", remote) if remote else None,
        "commit": _git_commit(folder),
    }


def package_commit() -> str | None:
    """Commit of the installed package, from the record pip and uv keep of where it was installed from."""
    try:
        origin = metadata.distribution("ua-tts-eval").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not origin:
        return None
    origin = json.loads(origin)
    if "vcs_info" in origin:
        return origin["vcs_info"]["commit_id"][:7]
    # Only an editable install sits in this package's own checkout. A regular one is usually
    # inside the model's clone, where git would report the model's commit.
    if origin.get("dir_info", {}).get("editable"):
        return _git_commit(Path(__file__).parent)
    return None


def _cpu() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return _run("sysctl", "-n", "machdep.cpu.brand_string") or platform.processor() or None


def _gpu() -> str | None:
    names = _run("nvidia-smi", "--query-gpu=name", "--format=csv,noheader")
    return ", ".join(names.splitlines()) if names else None


def _ram_gb() -> int | None:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3)
    except (AttributeError, OSError, ValueError):
        return None


def hardware() -> dict:
    return {
        "cpu": _cpu(),
        "gpu": _gpu(),
        "ram_gb": _ram_gb(),
        "os": f"{platform.system()} {platform.release()}",
    }
