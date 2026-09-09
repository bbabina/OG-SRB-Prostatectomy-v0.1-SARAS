r"""Apply the consolidated OG-SRB v0.2.1 methodological patch.

Usage:
    python APPLY_PATCH.py --repo "C:\path\to\OG-SRB-Prostatectomy-v0.1-SARAS"
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    package = Path(__file__).resolve().parent
    patch_files = package / "patch_files"

    required = [
        repo / "ontology" / "ogsrb_prostatectomy_v0.2.yaml",
        repo / "eval",
        repo / "baselines",
        repo / "reports",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(
            "This does not look like the expected repository. Missing:\n- "
            + "\n- ".join(missing)
        )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_root = repo / "patch_backups" / f"before-v0.2.1-{stamp}"

    copied = []
    backed_up = []

    for src in patch_files.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(patch_files)
        dst = repo / rel

        if dst.exists():
            backup = backup_root / rel
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst, backup)
            backed_up.append(rel)

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel)

    print("Consolidated patch applied successfully.")
    print(f"Repository: {repo}")
    print(f"Files installed/updated: {len(copied)}")
    if backed_up:
        print(f"Backup: {backup_root}")

    print("\nNEXT:")
    print(f'  cd "{repo}"')
    print("  python benchmarks/create_test180_manifest.py")
    print("\nThen follow RUN_ME.md from the patch package.")
    print("\nDo NOT apply any earlier ontology/decoder patch after this one.")


if __name__ == "__main__":
    main()
