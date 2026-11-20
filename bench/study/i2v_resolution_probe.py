"""Probe: can the local models render the 9:16 bucket OpenMontage actually uses?

The earlier Tier 3a round declared CogVideoX-5B-I2V unavailable because diffusers
raised "It is currently not possible to generate videos at a different
resolution". Reading the guard in ``diffusers/models/embeddings.py`` shows what
it actually enforces: a *learned* positional embedding must match the
checkpoint's own ``sample_height``/``sample_width``. CogVideoX-5B-I2V ships
60x90 latents, which is 480x720 pixels — portrait 9:16, the bucket OpenMontage
defaults to. So the model was never broken; the probe was asking it for 832x480.

This script measures what is actually true:

  1. cogvideox5b (T2V)      at 480x720
  2. cogvideox5b_i2v        at 480x720 with a real keyframe
  3. cogvideox5b (T2V)      at 832x480, as the control that must still work

It writes PNG/MP4 pairs under /tmp and prints one line per case. Nothing is
judged here; the question is only whether the pipeline runs and how long it
takes.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path("/mdata/shiweisong/VideoGenerationAgent")
sys.path.insert(0, str(ROOT))

PORTRAIT = (720, 480)   # height, width  -> 9:16
LANDSCAPE = (480, 832)  # height, width  -> 16:9 control


def keyframe(path: Path) -> Path:
    """A portrait still for the i2v case, drawn by the image model we already have."""
    import torch
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        str(ROOT / "bench/models/hf/stable-diffusion-v1-5/stable-diffusion-v1-5"),
        torch_dtype=torch.float16, safety_checker=None,
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.to("cuda")
    img = pipe(
        prompt="cinematic still, a delivery courier in a dark grey jacket standing in a rainy alley, "
               "square face, short buzz cut, scar through the left eyebrow, medium shot",
        height=720, width=480, num_inference_steps=25, guidance_scale=7.5,
        generator=torch.Generator(device="cuda").manual_seed(0),
    ).images[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    del pipe
    torch.cuda.empty_cache()
    return path


def run_case(label: str, model: str, size: tuple[int, int], image: Path | None) -> None:
    import torch
    from diffusers import DiffusionPipeline
    from diffusers.utils import export_to_video

    spec_path = {
        "cogvideox5b": "zai-org/CogVideoX-5b",
        "cogvideox5b_i2v": "zai-org/CogVideoX-5b-I2V",
    }[model]
    root = ROOT / "bench/models/hf" / spec_path
    height, width = size

    t0 = time.monotonic()
    pipe = DiffusionPipeline.from_pretrained(str(root), torch_dtype=torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)
    pipe.to("cuda")
    load_s = time.monotonic() - t0

    kwargs = dict(
        prompt="a courier walks through a rainy alley, camera follows",
        num_frames=41, height=height, width=width,
        num_inference_steps=40, guidance_scale=6.0,
        generator=torch.Generator(device="cpu").manual_seed(0),
    )
    if image is not None:
        from PIL import Image

        kwargs["image"] = Image.open(image).convert("RGB").resize((width, height))

    t1 = time.monotonic()
    try:
        out = pipe(**kwargs)
    except Exception as exc:  # noqa: BLE001 - the answer is the error text
        print(f"{label:34s} FAILED  {type(exc).__name__}: {str(exc)[:150]}", flush=True)
        del pipe
        torch.cuda.empty_cache()
        return
    render_s = time.monotonic() - t1

    target = Path(f"/tmp/i2v_probe/{label.replace(' ', '_')}.mp4")
    target.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(out.frames[0], str(target), fps=8)
    print(f"{label:34s} OK  load {load_s:5.1f}s  render {render_s:6.1f}s  "
          f"{width}x{height}  {out.frames[0][0].size}", flush=True)

    del pipe
    torch.cuda.empty_cache()


def main() -> int:
    still = keyframe(Path("/tmp/i2v_probe/keyframe.png"))
    print(f"keyframe: {still} ({still.stat().st_size // 1024} KB)", flush=True)

    run_case("t2v 9:16 480x720", "cogvideox5b", PORTRAIT, None)
    run_case("i2v 9:16 480x720", "cogvideox5b_i2v", PORTRAIT, still)
    run_case("t2v 16:9 832x480 (control)", "cogvideox5b", LANDSCAPE, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
