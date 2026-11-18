#!/usr/bin/env python3
"""How much of the approved budget this study has actually spent.

Reads the token counts each episode recorded in its own state file and applies
the documented Seedance rate, rather than assuming a per-episode price. The
first estimate I made from duration alone predicted 27.74 CNY for a 15-shot
episode; the token records agree, but they are the evidence and the estimate is
only a check.

Scope is deliberately the current funded run. Projects from before the top-up
are listed separately: they consumed the *previous* balance, and folding them
into this total would make the remaining budget look far smaller than it is.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/mdata/shiweisong/VideoGenerationAgent")
RATE = 37.0 / 1e6          # CNY per token, documented in bench/backends.py
LIMIT = 100.0              # approved top-up
# Episodes rendered after the 2026-11-17 20:40 top-up. Anything older was paid
# from the previous balance.
#
# A hardcoded list was wrong: it named only the first funded episode, so the
# second one's 20 CNY of rendering was invisible and the remaining budget read
# 21 CNY too high. Funding is now decided by file mtime — a project directory
# created after the top-up is part of this run — with the roster still listed so
# a reader can see exactly what was summed.
FUNDED_SINCE = 1794931200.0          # 2026-11-17 20:40 local, top-up time
FUNDED = {"cb6675e5", "3a6dcd2d"}    # current run, listed for transparency


def is_funded(project_dir: Path) -> bool:
    return project_dir.stat().st_mtime >= FUNDED_SINCE or project_dir.name.replace("project_", "") in FUNDED


def cost_of(project_id: str) -> dict:
    state_path = ROOT / "workspace/projects" / f"project_{project_id}" / "state.json"
    if not state_path.exists():
        return {"id": project_id, "video_tokens": 0, "image_tokens": 0, "cny": 0.0}
    state = json.loads(state_path.read_text(encoding="utf-8"))
    media = (state.get("usage") or {}).get("media") or {}
    video = media.get("generate_video", {}).get("tokens", 0)
    image = media.get("generate_image", {}).get("tokens", 0)
    return {"id": project_id, "video_tokens": video, "image_tokens": image,
            "cny": round((video + image) * RATE, 2)}


def om_cost(project_id: str) -> float:
    """OpenMontage records no tokens; cost it from rendered seconds.

    One Seedance clip per shot at its planned duration, which is what the
    studio submits. Stated as an estimate rather than measured, because the
    studio keeps no usage ledger.
    """
    state_path = ROOT / "bench/_vendor/OpenMontage/bench_run/projects" / project_id / "studio_state.json"
    if not state_path.exists():
        return 0.0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    shots = [s for s in (state.get("shots") or []) if not s.get("deletedAt")]
    usd_seconds = 0.0
    for shot in shots:
        for take in shot.get("videos") or []:
            usd_seconds += float(take.get("duration") or 0)
    # 480p fast: ~CNY 0.372 per second (4 s = 1.49), from the measured 4 s/5 s/8 s
    # points in bench/study/README.md.
    return round(usd_seconds * 0.372, 2)


def main(argv: list[str]) -> int:
    rows = [cost_of(p.replace("project_", "")) for p in
            [d.name for d in (ROOT / "workspace/projects").glob("project_*")]]
    funded = [r for r in rows if is_funded(ROOT / "workspace/projects" / f"project_{r['id']}")]
    legacy = [r for r in rows if not is_funded(ROOT / "workspace/projects" / f"project_{r['id']}") and r["cny"] > 0]

    ours_spent = sum(r["cny"] for r in funded)
    # OpenMontage projects created after the top-up.
    om_new = ["benchs2"]
    om_spent = sum(om_cost(p) for p in om_new)

    print(f"预算上限         {LIMIT:7.2f} CNY")
    print(f"本项目（本轮）   {ours_spent:7.2f} CNY")
    for r in funded:
        print(f"   {r['id']}  video {r['video_tokens']:>7} tok  image {r['image_tokens']:>6} tok  {r['cny']:6.2f}")
    print(f"OpenMontage（本轮）{om_spent:6.2f} CNY  （按秒估算，studio 无 token 记录）")
    total = ours_spent + om_spent
    print(f"{'合计':16s} {total:7.2f} CNY   余量 {LIMIT - total:6.2f}")
    if legacy:
        print()
        print(f"充值前已消耗（不计入本轮）: {sum(r['cny'] for r in legacy):.2f} CNY / {len(legacy)} 个 episode")
    if total > LIMIT:
        print("\n超支！")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
