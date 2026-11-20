"""Download the paper's GQCBM checkpoints from Hugging Face into ckpts/.

Checkpoints are hosted on Hugging Face, not in git:
https://huggingface.co/the-matter-lab/GQCBM (files under ckpts/).

Usage (from the repository root):
    pip install huggingface_hub
    python scripts/download_ckpts.py                 # all checkpoints, pinned revision
    python scripts/download_ckpts.py --revision main # latest upload
    python scripts/download_ckpts.py gqe_bimodal_20q extras_bas4   # selected files

Downloaded files are checked against MANIFEST.sha256 from the same revision.
"""
import argparse
import hashlib
import os
import sys

from huggingface_hub import hf_hub_download, snapshot_download

REPO_ID = 'the-matter-lab/GQCBM'
# Hugging Face commit holding every checkpoint the paper uses (29 files, see REPRODUCE.md).
DEFAULT_REVISION = 'f3d35c17183b0cb3792cfd1488f5f90b83ddcff7'
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('names', nargs='*',
                    help='checkpoint names without .pt (default: all)')
    ap.add_argument('--revision', default=DEFAULT_REVISION)
    args = ap.parse_args()

    patterns = ([f'ckpts/{n}.pt' for n in args.names] if args.names
                else ['ckpts/*.pt'])
    snapshot_download(REPO_ID, revision=args.revision, local_dir=REPO_ROOT,
                      allow_patterns=patterns + ['MANIFEST.sha256'])

    manifest = os.path.join(REPO_ROOT, 'MANIFEST.sha256')
    expected = dict(line.split()[::-1] for line in open(manifest) if line.strip())
    bad = 0
    for rel, digest in sorted(expected.items()):
        path = os.path.join(REPO_ROOT, rel)
        if not os.path.exists(path):
            continue  # not requested
        if sha256(path) != digest:
            print(f'SHA-256 MISMATCH: {rel}')
            bad += 1
        else:
            print(f'ok  {rel}')
    os.remove(manifest)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
