"""Vendor Video-Bench at a pinned commit into bench/_vendor.

The checkout is deliberately not committed: it is 27 MB including its own
``.git``, and a nested repository inside this one makes ``git status`` noisy and
turns every upstream update into a conflict. Pinning the commit here instead
means a run's numbers can be traced to exact upstream source.

The pinned commit is the one this harness was validated against. Its
``VideoTextAlignment.eval`` is broken (see ``bench/judge.py``), which is why the
harness runs its own loop rather than calling upstream's; the vendored copy is
used for its prompts, dimension definitions and the 419-prompt suite.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/Video-Bench/Video-Bench.git"
# Last upstream commit that changed videobench/VideoTextAlignment.py.
PINNED = "d40b917c4238011aea27d24f85b1e3bf996715bd"

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "_vendor" / "Video-Bench"


def main() -> int:
    if (TARGET / "videobench" / "prompt_dict.py").exists():
        head = subprocess.run(["git", "-C", str(TARGET), "rev-parse", "HEAD"],
                              capture_output=True, text=True)
        print(f"already vendored at {head.stdout.strip()[:12] or 'unknown'}")
        return 0
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", REPO, str(TARGET)], check=True)
    subprocess.run(["git", "-C", str(TARGET), "checkout", "--detach", PINNED], check=True)
    print(f"vendored Video-Bench at {PINNED[:12]} -> {TARGET}")
    print("The prompts, dimension scales and VideoBench_full.json come from here; "
          "the scoring loop lives in bench/judge.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
