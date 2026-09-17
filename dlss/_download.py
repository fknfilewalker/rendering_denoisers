"""Fetch the DLSS Ray Reconstruction snippet from NVIDIA's GitHub repository.

The display driver only ships the NGX *loader*; the feature itself
(``nvngx_dlssd.dll`` on Windows, ``libnvidia-ngx-dlssd.so.*`` on Linux) is a
separate binary that NGX loads from disk, and NVIDIA publishes it in
https://github.com/NVIDIA/DLSS. This module downloads it on first use and caches
it, so that nothing has to be vendored into this repository.

The binary is NVIDIA's, under the licence in that repository (LICENSE.txt);
downloading it here means accepting those terms. Nothing is fetched when
``MI_DLSS_LIBRARY_PATH`` already points at a directory holding the snippet, or
when ``download=False`` is passed to the denoiser.

The default is a pinned version, so a given checkout keeps fetching the same
binary; pass ``version="latest"`` to follow the newest tag instead.
"""

import hashlib
import json
import os
import pathlib
import platform
import shutil
import sys
import tempfile
import urllib.request

REPO = "NVIDIA/DLSS"
DEFAULT_VERSION = "v310.9.1"
_API = "https://api.github.com"

# The snippet is named after the feature; on Linux the version is part of the
# file name, so the exact name is taken from the directory listing.
_PLATFORMS = {
    ("win32", "amd64"):   ("Windows_x86_64", "nvngx_dlssd.dll"),
    ("win32", "x86_64"):  ("Windows_x86_64", "nvngx_dlssd.dll"),
    ("win32", "arm64"):   ("Windows_aarch64", "nvngx_dlssd.dll"),
    ("win32", "aarch64"): ("Windows_aarch64", "nvngx_dlssd.dll"),
    ("linux", "x86_64"):  ("Linux_x86_64", "libnvidia-ngx-dlssd.so"),
    ("linux", "amd64"):   ("Linux_x86_64", "libnvidia-ngx-dlssd.so"),
    ("linux", "aarch64"): ("Linux_aarch64", "libnvidia-ngx-dlssd.so"),
    ("linux", "arm64"):   ("Linux_aarch64", "libnvidia-ngx-dlssd.so"),
}


class DownloadError(RuntimeError):
    pass


def target() -> tuple[str, str]:
    """The (repository directory, file name prefix) for this machine."""
    key = (sys.platform, platform.machine().lower())
    if key not in _PLATFORMS:
        raise DownloadError(f"DLSS ships no Ray Reconstruction binary for {key[0]}/{key[1]}")
    return _PLATFORMS[key]


def cache_directory() -> pathlib.Path:
    """Where downloaded snippets are kept, overridable with ``RTM_DLSS_CACHE``."""
    if (env := os.environ.get("RTM_DLSS_CACHE")) is not None:
        return pathlib.Path(env)
    if sys.platform == "win32":
        base = pathlib.Path(os.environ.get("LOCALAPPDATA", pathlib.Path.home() / "AppData/Local"))
    elif sys.platform == "darwin":
        base = pathlib.Path.home() / "Library/Caches"
    else:
        base = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache"))
    return base / "rtm-dlss"


def _request(url: str) -> urllib.request.Request:
    request = urllib.request.Request(url, headers={"User-Agent": "rtm-denoiser"})
    if (token := os.environ.get("GITHUB_TOKEN")):
        request.add_header("Authorization", f"Bearer {token}")
    return request


def latest_version() -> str:
    """The newest tag of the DLSS repository."""
    with urllib.request.urlopen(_request(f"{_API}/repos/{REPO}/tags?per_page=1")) as response:
        tags = json.load(response)
    if not tags:
        raise DownloadError(f"{REPO} has no tags")
    return tags[0]["name"]


def _listing(version: str, variant: str) -> list[dict]:
    directory, _ = target()
    url = f"{_API}/repos/{REPO}/contents/lib/{directory}/{variant}?ref={version}"
    try:
        with urllib.request.urlopen(_request(url)) as response:
            return json.load(response)
    except Exception as e:
        raise DownloadError(f"could not list lib/{directory}/{variant} at {version}: {e}") from e


def _blob_sha1(path: pathlib.Path) -> str:
    """The SHA-1 git would give this file, which is what the API reports."""
    digest = hashlib.sha1(b"blob %d\0" % path.stat().st_size)
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _notice(version: str):
    """Say whose binary this is, and under what terms, before fetching it.

    The snippet is not part of this package and is never redistributed with it:
    it comes from NVIDIA, and the licence is accepted by whoever runs the
    download, on their own machine. Printed unconditionally, since a licence
    notice should not depend on ``progress``.
    """
    for line in (f"fetching the Ray Reconstruction snippet from {REPO} {version}.",
                 "The binary is NVIDIA's, not part of this package; using it",
                 f"means accepting https://github.com/{REPO}/blob/{version}/LICENSE.txt"):
        print(f"[dlss] {line}", file=sys.stderr)


def _fetch(entry: dict, destination: pathlib.Path, progress: bool):
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = entry.get("size", 0)
    handle, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".part")
    temporary = pathlib.Path(temporary)
    try:
        with urllib.request.urlopen(_request(entry["download_url"])) as response, \
                os.fdopen(handle, "wb") as out:
            done = 0
            while block := response.read(1 << 20):
                out.write(block)
                done += len(block)
                if progress and total:
                    print(f"\r[dlss] downloading {destination.name}: "
                          f"{100 * done // total:3d}% of {total / 2**20:.0f} MiB",
                          end="", file=sys.stderr, flush=True)
        if progress and total:
            print(file=sys.stderr)
        if (sha := _blob_sha1(temporary)) != entry["sha"]:
            raise DownloadError(f"{destination.name} is corrupt: expected blob {entry['sha']}, "
                                f"got {sha}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def library(version: str | None = None, *, variant: str = "rel",
            directory: str | os.PathLike | None = None, progress: bool = True) -> pathlib.Path:
    """The directory holding the Ray Reconstruction snippet, downloading it if needed.

    Args:
        version: A tag of https://github.com/NVIDIA/DLSS, or ``"latest"``.
            Defaults to the pinned :data:`DEFAULT_VERSION`.
        variant: ``"rel"`` for the release build, ``"dev"`` for the one with the
            on-screen debug overlay.
        directory: Where to keep the download. Defaults to a per-version folder
            under :func:`cache_directory`.
        progress: Print download progress to stderr.

    Returns:
        The directory to hand to NGX as its search path.
    """
    if version is None:
        version = DEFAULT_VERSION
    if version == "latest":
        version = latest_version()
    _, prefix = target()
    directory = pathlib.Path(directory) if directory is not None \
        else cache_directory() / version / variant

    if directory.is_dir() and any(f.name.startswith(prefix) for f in directory.iterdir()):
        return directory

    entries = [e for e in _listing(version, variant) if e["name"].startswith(prefix)]
    if not entries:
        raise DownloadError(f"{REPO} {version} has no {prefix}* for this platform")
    _notice(version)
    _fetch(entries[0], directory / entries[0]["name"], progress)
    return directory


def resolve(library_path: str | os.PathLike | None, *, download: bool = True,
            version: str | None = None, progress: bool = True) -> pathlib.Path | None:
    """Work out the NGX search path: an explicit one, the environment, or a download.

    Returns ``None`` when nothing was given and downloading is off, which leaves
    NGX to search wherever it would by default.
    """
    if library_path is not None:
        return pathlib.Path(library_path)
    if os.environ.get("MI_DLSS_LIBRARY_PATH"):
        return None  # ngx.Common picks the environment variable up itself
    if not download:
        return None
    return library(version, progress=progress)


def clear_cache():
    """Delete every downloaded snippet."""
    shutil.rmtree(cache_directory(), ignore_errors=True)

