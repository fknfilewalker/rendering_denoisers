"""Download the DLSS Ray Reconstruction snippet, ahead of its first use."""

import argparse

from ._download import cache_directory, clear_cache, library

parser = argparse.ArgumentParser(
    prog="python -m rendering_denoisers.dlss",
    description="Download the NVIDIA DLSS Ray Reconstruction snippet.")
parser.add_argument("--version", default=None, help='tag of NVIDIA/DLSS, or "latest"')
parser.add_argument("--variant", default="rel", choices=("rel", "dev"),
                    help="release build, or the one with the debug overlay")
parser.add_argument("--directory", default=None, help="where to put the snippet")
parser.add_argument("--clear", action="store_true", help="delete the cache and exit")
args = parser.parse_args()

if args.clear:
    clear_cache()
    print(f"removed {cache_directory()}")
else:
    print(library(args.version, variant=args.variant, directory=args.directory))
