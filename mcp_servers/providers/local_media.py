"""Local video generation: the open-weight models already on this disk.

Free to run and free to repeat, which is the point: every paid comparison in
this repo is capped at a handful of samples by its budget, and the cost of a
free layer is wall-clock only.

Three text-to-video models and, when its weights are present, one
image-to-video model. Weights are read from ``bench/models`` and never fetched
at generation time — a provider that quietly downloads 20 GB in the middle of a
run would make the run's identity unreproducible.

The model is named by ``VIDEO_MODEL``; which frame roles the resulting instance
consumes is declared on the instance (``frame_roles``), so the orchestration
never has to match on a model id.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from mcp_servers.providers.base import VideoRequest

# Every local model renders one fixed shape here. 832x480 landscape is the only
# resolution all three share: CogVideoX and Wan will accept either orientation,
# but the vendored LongLive writes its latent noise as [N, 16, 60, 104], which
# is 480x832 *landscape* (60*8 by 104*8). Nothing reads config.height/width
# there, so it cannot be asked for portrait without editing the vendored tree.
#
# Measured, not assumed: CogVideoX at 480x832 and at 832x480 both render, as
# does Wan at 480x720 and 832x480. Landscape is the intersection.
LOCAL_WIDTH = 832
LOCAL_HEIGHT = 480

# Per-model render contract. The two families disagree on both frame rate and
# on how a frame count must be shaped, and feeding one the other's numbers
# produces a clip of the wrong length rather than an error.
MODELS: dict[str, dict] = {
    "cogvideox5b": {
        "kind": "diffusers",
        "path": "zai-org/CogVideoX-5b",
        "fps": 8,
        "frame_rule": "4k+1",
        "steps": 40,
        "guidance": 6.0,
        # CogVideoXPipeline takes text only. Declaring a first_frame role here
        # would send the run through the keyframe stage to produce pictures the
        # model silently ignores.
        "roles": frozenset(),
    },
    "wan21_1_3b": {
        "kind": "diffusers",
        "path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "fps": 16,
        "frame_rule": "4k+1 or 81",
        "steps": 40,
        "guidance": 5.0,
        # WanPipeline is text-to-video as well; its image variant is a separate
        # model (Wan2.1-I2V), which is not what is loaded here.
        "roles": frozenset(),
    },
    "longlive_1_3b": {
        "kind": "longlive",
        "fps": 16,
        "steps": 0,
        "guidance": 0.0,
        "roles": frozenset(),
    },
    # The one image-to-video model here. Unlike the three above it consumes a
    # first frame, so its ``roles`` is non-empty and the orchestration keeps the
    # conditioning path instead of skipping it.
    "cogvideox5b_i2v": {
        "kind": "diffusers",
        "path": "zai-org/CogVideoX-5b-I2V",
        "fps": 8,
        "frame_rule": "4k+1",
        "steps": 40,
        "guidance": 6.0,
        "roles": frozenset({"first_frame"}),
        "i2v": True,
    },
}

T2V_MODELS = frozenset({"cogvideox5b", "wan21_1_3b", "longlive_1_3b"})


def _bench_root() -> Path:
    """``bench/`` regardless of where the caller's cwd is."""
    return Path(__file__).resolve().parents[2] / "bench"


def _frames_for(rule: str, fps: int, seconds: float) -> int:
    """Frame count the model will accept for roughly ``seconds``.

    CogVideoX wants a 4k+1 count; Wan accepts any 4k+1 up to 81; LongLive counts
    latent frames, which ``_longlive_latent`` handles separately because its
    VAE decodes N latents to 4*(N-1)+1 pixels.
    """
    raw = int(round(seconds * fps))
    if rule == "4k+1":
        return max(9, (raw // 4) * 4 + 1)
    if rule == "4k+1 or 81":
        return min(81, max(9, (raw // 4) * 4 + 1))
    return max(9, raw)


class LocalVideoProvider:
    """Runs an open-weight video model on this host's GPU.

    Return shape matches the Ark provider so nothing downstream needs a special
    case: ``video_path`` relative to the project directory, ``provider``, and a
    ``usage`` that says plainly where the money went — nowhere.
    """

    name = "local"

    def __init__(self, model: str, root: Path | None = None) -> None:
        if model not in MODELS:
            raise ValueError(
                f"VIDEO_MODEL={model!r} is not a local model; choose from {', '.join(sorted(MODELS))}"
            )
        self.model = model
        self.spec = MODELS[model]
        self.model_root = (root or (_bench_root() / "models" / "hf"))
        self._pipe = None

    @property
    def frame_roles(self) -> frozenset[str]:
        return frozenset(self.spec["roles"])

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def _diffusers_path(self) -> Path:
        path = self.model_root / self.spec["path"]
        if not (path / "model_index.json").exists():
            raise FileNotFoundError(
                f"{self.model} weights are not at {path}; fetch them with "
                f"bench/fetch_models.sh {self.spec['path']} rather than letting the run download them"
            )
        return path

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        import torch
        from diffusers import DiffusionPipeline

        pipe = DiffusionPipeline.from_pretrained(str(self._diffusers_path()), torch_dtype=torch.bfloat16)
        pipe.set_progress_bar_config(disable=True)
        pipe.to("cuda")
        self._pipe = pipe
        return pipe

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def generate(self, request: VideoRequest) -> dict:
        started = time.monotonic()
        if self.spec["kind"] == "longlive":
            produced, frames = self._generate_longlive(request)
        else:
            produced, frames = self._generate_diffusers(request)
        elapsed = time.monotonic() - started

        target = request.project_dir / "videos" / f"{request.shot_id}.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)
        if produced != target:
            produced.replace(target)

        return {
            "video_path": str(target.relative_to(request.project_dir)),
            "provider": self.name,
            "model": self.model,
            "motion_prompt": request.motion_prompt,
            "duration_s": frames / self.spec["fps"],
            "frames": frames,
            "fps": self.spec["fps"],
            "resolution": f"{LOCAL_WIDTH}x{LOCAL_HEIGHT}",
            "elapsed_s": round(elapsed, 1),
            # Local inference bills nothing. Kept as an explicit empty map
            # rather than a zeroed token count so a report cannot mistake it
            # for a metered call that happened to cost nothing.
            "usage": {},
            "billed": False,
        }

    def _generate_diffusers(self, request: VideoRequest) -> tuple[Path, int]:
        import torch
        from diffusers.utils import export_to_video

        pipe = self._load()
        frames = _frames_for(self.spec["frame_rule"], self.spec["fps"], request.duration_s)
        generator = torch.Generator(device="cpu").manual_seed(request.seed)

        kwargs = {
            "prompt": request.motion_prompt,
            "num_frames": frames,
            "height": LOCAL_HEIGHT,
            "width": LOCAL_WIDTH,
            "num_inference_steps": self.spec["steps"],
            "guidance_scale": self.spec["guidance"],
            "generator": generator,
        }
        # An i2v model takes the keyframe as an image argument; the same class
        # serves both kinds because the only difference is this keyword.
        if self.spec.get("i2v") and request.image_path is not None:
            from PIL import Image

            kwargs["image"] = Image.open(request.image_path).convert("RGB").resize((LOCAL_WIDTH, LOCAL_HEIGHT))

        output = pipe(**kwargs)
        target = request.project_dir / "videos" / f"{request.shot_id}.raw.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(output.frames[0], str(target), fps=self.spec["fps"])
        return target, frames

    def _generate_longlive(self, request: VideoRequest) -> tuple[Path, int]:
        """Drive the vendored LongLive repo as a subprocess.

        Its inference script owns model loading, so it is invoked rather than
        imported. The latent count is rounded up to a multiple of
        ``num_frame_per_block`` (3) and decoded as 4*(N-1)+1 pixel frames —
        passing a pixel count here yields a clip about 4x too long.
        """
        import shutil

        import yaml

        bench = _bench_root()
        repo = bench / "_vendor" / "LongLive" / "LongLive1.0"
        if not (repo / "inference.py").exists():
            raise FileNotFoundError(f"LongLive is not vendored at {repo}")

        work = (request.project_dir / "videos" / f".longlive_{request.shot_id}").resolve()
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)

        block = 3
        target_frames = int(round(request.duration_s * self.spec["fps"]))
        latent = max(block, int(round((target_frames + 3) / 4 / block)) * block)
        pixel_frames = 4 * (latent - 1) + 1

        prompt_file = work / "prompts.txt"
        prompt_file.write_text(request.motion_prompt + "\n", encoding="utf-8")

        config = yaml.safe_load((repo / "configs" / "longlive_inference.yaml").read_text(encoding="utf-8"))
        config.update(
            data_path=str(prompt_file),
            output_folder=str(work / "clips"),
            num_output_frames=latent,
            num_samples=1,
            seed=request.seed,
            save_with_index=True,
        )
        config_path = work / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        proc = subprocess.run(
            [str(Path(__import__("sys").executable)), "-u", "inference.py", "--config_path", str(config_path)],
            cwd=str(repo), capture_output=True, text=True, timeout=3600,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-12:]
            raise RuntimeError("LongLive failed:\n" + "\n".join(tail))

        produced = sorted((work / "clips").glob("*.mp4"))
        if not produced:
            raise RuntimeError(f"LongLive produced no video in {work / 'clips'}")
        return produced[0], pixel_frames
