"""Video generators the benchmark compares.

Each backend answers one question: given a text prompt and a seed, produce an N
second vertical clip. How it gets there is the backend's business, and the
difference is recorded rather than hidden — the pipeline generates a keyframe
first and animates it, while the open-source models go straight from text. A
report that silently compared image-to-video against text-to-video would be
measuring the wrong thing, so ``Backend.modality`` travels with every result.

Backends are lazy: importing this module must not import torch, because the
judge-only path (score existing clips) has no reason to load a 5B model.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .config import LONGLIVE_ROOT, MODEL_ROOT, OPENMONTAGE_ROOT, PROJECT_ROOT, BenchConfig


@dataclass
class Generation:
    """One clip produced for the benchmark."""

    model: str
    prompt: str
    seed: int
    path: Path
    seconds: float
    width: int
    height: int
    fps: int
    modality: str
    elapsed_s: float
    note: str = ""


class Backend(Protocol):
    name: str
    modality: str

    def available(self) -> tuple[bool, str]: ...

    def generate(self, prompt: str, seed: int, seconds: float, out_dir: Path) -> Generation: ...


# ── our pipeline ──────────────────────────────────────────────────────────


class PipelineBackend:
    """The project's own generator, driven at shot level.

    A fair single-shot comparison has to use the same stage the others occupy,
    so this renders one shot rather than a whole episode: keyframe (Seedream)
    then clip (Seedance). That is image-to-video with an authored first frame,
    which is what ``modality`` reports. It is also the expensive path, and the
    only one that spends Ark money.

    With ``IMAGE_PROVIDER=mock`` the keyframe is a locally drawn placeholder and
    nothing is billed; the clip then comes from whatever ``VIDEO_PROVIDER`` is.
    """

    name = "ours"
    modality = "i2v (keyframe -> Seedance)"

    def __init__(self, config: BenchConfig):
        self.config = config

    def available(self) -> tuple[bool, str]:
        import os

        provider = os.environ.get("VIDEO_PROVIDER", "mock")
        if provider == "mock":
            return False, "VIDEO_PROVIDER=mock renders a placeholder clip, which is not a competitor"
        if provider == "ark" and os.environ.get("ARK_ALLOW_PAID") != "1":
            return False, "VIDEO_PROVIDER=ark needs ARK_ALLOW_PAID=1"
        # The keyframe is the first frame Seedance animates, so a placeholder
        # keyframe puts burn-in text into the final clip. This happened: a run
        # exported VIDEO_PROVIDER=ark but not IMAGE_PROVIDER, so the keyframe
        # fell back to mock and the clip shipped with "MOCK FRAME · replace
        # IMAGE_PROVIDER to render real art" across the frame — and the judge
        # scored it 5/5/5/5, because the text is legible and the picture is
        # sharp. A mixed-provider clip is not this pipeline's output, so it is
        # refused rather than scored.
        image_provider = os.environ.get("IMAGE_PROVIDER", "mock")
        if image_provider == "mock":
            return (
                False,
                "IMAGE_PROVIDER=mock draws a placeholder keyframe, which burns its "
                "label into the clip Seedance animates; set IMAGE_PROVIDER=ark too",
            )
        return True, f"video={provider} image={image_provider}"

    def generate(self, prompt: str, seed: int, seconds: float, out_dir: Path) -> Generation:
        import os
        import shutil
        import time

        from mcp_servers.video_tools_server import generate_video, ensure_project_dirs
        from mcp_servers.providers import get_image_provider, reset_provider_cache
        from mcp_servers.providers.base import ImageRequest

        root = ensure_project_dirs(f"bench_{out_dir.name}")
        started = time.monotonic()
        media = __import__("agent.settings", fromlist=["get_settings"]).get_settings().media

        image = get_image_provider().generate(
            ImageRequest(
                prompt=prompt,
                negative_prompt="",
                shot_id=f"shot_bench_{seed}",
                project_dir=root,
                width=media.width,
                height=media.height,
                seed=seed,
            )
        )
        result = generate_video(
            image["image_path"],
            prompt,
            f"bench_{out_dir.name}",
            f"shot_bench_{seed}",
            duration_s=seconds,
            seed=seed,
        )
        elapsed = time.monotonic() - started
        produced = root / result["video_path"]
        target = out_dir / f"{self.name}_{seed}.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(produced, target)
        reset_provider_cache()
        return Generation(
            model=self.name,
            prompt=prompt,
            seed=seed,
            path=target,
            seconds=seconds,
            width=media.width,
            height=media.height,
            fps=media.fps,
            modality=self.modality,
            elapsed_s=elapsed,
            note=f"image_provider={media.image_provider}",
        )


# ── open-source backends ──────────────────────────────────────────────────


class DiffusersT2VBackend:
    """Any diffusers text-to-video pipeline, by model id.

    Frames are chosen to land near ``seconds`` at the pipeline's own frame rate,
    because the two families do not agree: CogVideoX is 8fps and wants a 4k+1
    frame count, Wan is 16fps. Overriding one with the other's numbers produces
    a clip of the wrong length or a shape error, so each declares its own.
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        *,
        fps: int,
        frame_rule: str,
        width: int,
        height: int,
        steps: int = 40,
        guidance: float = 6.0,
    ):
        self.name = name
        self.model_id = model_id
        self.fps = fps
        self.frame_rule = frame_rule
        self.width = width
        self.height = height
        self.steps = steps
        self.guidance = guidance
        self.modality = f"t2v {width}x{height}@{fps}fps"
        self._pipe = None

    def available(self) -> tuple[bool, str]:
        try:
            import diffusers  # noqa: F401
            import torch
        except Exception as exc:  # noqa: BLE001
            return False, f"diffusers/torch unavailable: {exc}"
        if not torch.cuda.is_available():
            return False, "CUDA is not available"
        return True, self.model_id

    def _num_frames(self, seconds: float) -> int:
        raw = int(round(seconds * self.fps))
        if self.frame_rule == "4k+1":
            return max(9, (raw // 4) * 4 + 1)
        if self.frame_rule == "4k+1 or 81":
            # Wan accepts any 4k+1 up to 81; 81 is 5.06s at 16fps.
            return min(81, max(9, (raw // 4) * 4 + 1))
        return max(9, raw)

    def _local_path(self) -> Path | None:
        """The plain diffusers directory ``fetch_models.sh`` fills.

        The hub cache layout is not used: the downloader writes each repo as
        ``bench/models/hf/<org>/<repo>/`` so ``from_pretrained`` can be pointed
        at a local directory and never touches the network mid-run.
        """
        from .config import LONGLIVE_ROOT, MODEL_ROOT, OPENMONTAGE_ROOT, PROJECT_ROOT, BenchConfig

        candidate = BenchConfig().model_root / self.model_id
        return candidate if (candidate / "model_index.json").exists() else None

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        import torch
        from diffusers import DiffusionPipeline

        source = self._local_path()
        pipe = DiffusionPipeline.from_pretrained(
            str(source) if source else self.model_id,
            torch_dtype=torch.bfloat16,
            cache_dir=None if source else str(BenchConfig().model_root),
        )
        pipe.set_progress_bar_config(disable=True)
        # Whole pipeline on the GPU. The host has 275 GB of VRAM, so a 5B model
        # needs no offloading; enable_model_cpu_offload() would stream every
        # weight back over PCIe on each of the 40 denoise steps for no reason
        # and make a 5-second clip take minutes longer than it has to.
        pipe.to("cuda")
        self._pipe = pipe
        return pipe

    def generate(self, prompt: str, seed: int, seconds: float, out_dir: Path) -> Generation:
        import time

        import torch
        from diffusers.utils import export_to_video

        pipe = self._load()
        frames = self._num_frames(seconds)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        started = time.monotonic()
        output = pipe(
            prompt=prompt,
            num_frames=frames,
            height=self.height,
            width=self.width,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance,
            generator=generator,
            output_type="pil" if "CogVideo" in type(pipe).__name__ else "np",
        )
        elapsed = time.monotonic() - started
        result = output.frames[0]
        target = out_dir / f"{self.name}_{seed}.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(result, str(target), fps=self.fps)
        return Generation(
            model=self.name,
            prompt=prompt,
            seed=seed,
            path=target,
            seconds=frames / self.fps,
            width=self.width,
            height=self.height,
            fps=self.fps,
            modality=self.modality,
            elapsed_s=elapsed,
            note=f"{self.model_id} | {self.steps} steps",
        )


# A long-video generation workflow, not a single-clip text-to-video model.
#
# LongLive wraps Wan2.1-1.3B in an autoregressive student that keeps a KV cache
# across blocks, so it generates a continuous stream rather than one fixed clip.
# It is driven through a subprocess for two reasons that are not laziness:
#
#   1. ``inference.py`` is a script: it parses argv and builds the pipeline at
#      import time, with model paths hardcoded relative to its own directory.
#   2. Its requirements pin ``diffusers==0.31.0`` while this harness runs 0.42.1.
#      Importing it in-process would either force a downgrade that breaks the
#      CogVideoX/Wan backends or silently load a different scheduler.
#
# So the backend writes a prompt file, invokes the vendored script, and collects
# the mp4 it produces. Everything the harness needs afterwards — payloads,
# judging — works on the file exactly as it does for the diffusers backends.
class LongLiveBackend:
    name = "longlive_1_3b"
    modality = "t2v autoregressive (Wan2.1-1.3B + LongLive LoRA)"

    def __init__(self, repo: Path, models_root: Path, python: Path, *, fps: int = 16):
        self.repo = repo
        self.models_root = models_root
        self.python = python
        self.fps = fps

    def available(self) -> tuple[bool, str]:
        script = self.repo / "inference.py"
        if not script.exists():
            return False, f"LongLive is not vendored at {self.repo}"
        ckpt = self.repo / "longlive_models" / "models" / "longlive_base.pt"
        if not ckpt.exists():
            return False, f"missing checkpoint {ckpt}"
        if not (self.repo / "wan_models" / "Wan2.1-T2V-1.3B" / "config.json").exists():
            return False, "missing wan_models/Wan2.1-T2V-1.3B (config.json + weights)"
        return True, str(self.repo)

    def generate(self, prompt: str, seed: int, seconds: float, out_dir: Path) -> Generation:
        import shutil
        import subprocess
        import time
        import yaml

        out_dir.mkdir(parents=True, exist_ok=True)
        # Absolute, because the subprocess runs with cwd set to the vendored
        # repo: a relative path would resolve inside LongLive instead.
        work = (out_dir / f".longlive_{seed}").resolve()
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)

        # ``num_output_frames`` counts LATENT frames, not video frames. Wan's VAE
        # upsamples the temporal axis 4x, so N latent frames decode to
        # 4*(N-1)+1 pixel frames — measured: 78 latent -> 309 frames. Passing a
        # target video-frame count here would silently produce a clip ~4x too
        # long (it did: a 5-second request yielded 19.3s).
        #
        # The count must also divide by num_frame_per_block (3), which the
        # pipeline asserts. Rounding to the nearest multiple of 3 rather than
        # down matters: for a 5s target the candidates are latent 18 (69 frames,
        # 4.31s) and latent 21 (81 frames, 5.06s), and only the latter is close.
        block = 3
        target_frames = int(round(seconds * self.fps))
        latent = max(block, int(round((target_frames + 3) / 4 / block)) * block)
        video_frames = 4 * (latent - 1) + 1
        prompt_file = work / "prompts.txt"
        prompt_file.write_text(prompt + "\n", encoding="utf-8")

        config = yaml.safe_load((self.repo / "configs" / "longlive_inference.yaml").read_text(encoding="utf-8"))
        config.update(
            data_path=str(prompt_file),
            output_folder=str(work / "clips"),
            num_output_frames=latent,
            num_samples=1,
            seed=seed,
            save_with_index=True,
        )
        config_path = work / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        started = time.monotonic()
        proc = subprocess.run(
            [str(self.python), "-u", "inference.py", "--config_path", str(config_path)],
            cwd=str(self.repo), capture_output=True, text=True, timeout=3600,
        )
        elapsed = time.monotonic() - started
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-12:]
            raise RuntimeError("LongLive failed:\n" + "\n".join(tail))

        produced = sorted((work / "clips").glob("*.mp4"))
        if not produced:
            raise RuntimeError(f"LongLive produced no video in {work / 'clips'}")
        target = out_dir / f"{self.name}_{seed}.mp4"
        shutil.copy2(produced[0], target)

        return Generation(
            model=self.name,
            prompt=prompt,
            seed=seed,
            path=target,
            seconds=video_frames / self.fps,
            width=832,
            height=480,
            fps=self.fps,
            modality=self.modality,
            elapsed_s=elapsed,
            note=f"latent={latent} pixel={video_frames} | LongLive-1.3B lora",
        )


# A multi-stage drama workflow from an open-source project, driven through its
# own generation tool.
#
# OpenMontage runs script -> characters -> scenes -> storyboard -> shots ->
# voice -> edit through a seven-stage studio. What this backend calls is its
# direct Ark adapter, i.e. the *shot* stage only: prompt in, clip out, no
# planning layer in front of it.
#
# That is deliberate, and it is the point of the baseline. The pipeline's own
# orchestration (script writing, storyboard, keyframe generation, reference
# propagation) lives in its web service, which needs Postgres, Redis and MinIO
# to boot. Reaching the shot tool directly isolates one variable: the same
# Seedance model, asked with the same bare prompt our pipeline starts from,
# but without a keyframe or a reference image in front of it. Any gap between
# this and `ours` is the work the orchestration does, not a different model.
class OpenMontageBackend:
    name = "openmontage_shot"
    modality = "t2v (OpenMontage shot stage, Seedance via Ark)"

    def __init__(self, repo: Path, *, width: int = 720, height: int = 1280, fps: int = 24):
        self.repo = repo
        self.width = width
        self.height = height
        self.fps = fps

    def available(self) -> tuple[bool, str]:
        if not (self.repo / "tools" / "video" / "seedance_ark.py").exists():
            return False, f"OpenMontage is not vendored at {self.repo}"
        if not os.environ.get("ARK_API_KEY"):
            return False, "ARK_API_KEY is not set"
        return True, f"{self.repo} (shot stage only)"

    def generate(self, prompt: str, seed: int, seconds: float, out_dir: Path) -> Generation:
        import time

        sys.path.insert(0, str(self.repo))
        try:
            from tools.video.seedance_ark import SeedanceArkVideo
        finally:
            sys.path.pop(0)

        out_dir.mkdir(parents=True, exist_ok=True)
        target = (out_dir / f"{self.name}_{seed}.mp4").resolve()

        tool = SeedanceArkVideo()
        started = time.monotonic()
        # The model has to be pinned to the one this account has activated.
        # Left out, the tool falls back to its own default
        # (``doubao-seedance-2-0-260128``), which this account has not opened
        # — both seeds failed with ModelNotOpen — and, more importantly, which
        # is a different build from the one ``ours`` renders with. A comparison
        # that claims to hold the generator fixed has to actually hold it fixed,
        # so ARK_VIDEO_MODEL is passed through explicitly and the report prints
        # it. ``model_variant`` is left at "standard" because the variant only
        # picks the fallback; an explicit model id wins over it.
        model_id = os.environ.get("ARK_VIDEO_MODEL", "")
        self.model_id = model_id
        # The resolution is read from the same variable the pipeline reads, not
        # hardcoded. A hardcoded 720p here measured 704x1248 against `ours`
        # at 480x864 — 2.1x the pixels — so every visual dimension compared two
        # different output budgets and called the difference workflow quality.
        result = tool.execute({
            "operation": "text_to_video",
            "prompt": prompt,
            "duration": int(max(4, round(seconds))),
            "aspect_ratio": "9:16",
            "resolution": os.environ.get("ARK_VIDEO_RESOLUTION", "720p"),
            "output_path": str(target),
            "seed": seed,
            # Ark bills by actual usage; this figure only feeds the tool's own
            # budget accounting, and an id outside its price table makes the
            # tool refuse to submit rather than guess.
            "custom_price_cny_per_million_tokens": 37.0,
            **({"model": model_id} if model_id else {}),
        })
        elapsed = time.monotonic() - started

        if not result.success:
            raise RuntimeError(f"OpenMontage seedance_ark failed: {result.error}")
        produced = Path(result.data["output_path"])
        if produced != target and produced.exists():
            produced.replace(target)
        if not target.exists():
            raise RuntimeError(f"OpenMontage produced no file at {target}")

        return Generation(
            model=self.name,
            prompt=prompt,
            seed=seed,
            path=target,
            seconds=float(result.data.get("duration") or seconds),
            width=self.width,
            height=self.height,
            fps=self.fps,
            modality=self.modality,
            elapsed_s=elapsed,
            note=f"{result.data.get('model', '')} | shot stage only, no storyboard",
        )


# Vertical is the project's canvas, so the open-source models are asked for the
# tallest shape each one was actually trained on. CogVideoX was trained at
# 720x480; 480x720 is its published portrait bucket. Wan's is 832x480, so its
# portrait counterpart is 480x832.
REGISTRY: dict[str, DiffusersT2VBackend | LongLiveBackend | OpenMontageBackend] = {
    "cogvideox5b": DiffusersT2VBackend(
        "cogvideox5b",
        "zai-org/CogVideoX-5b",
        fps=8,
        frame_rule="4k+1",
        width=480,
        height=720,
        steps=40,
        guidance=6.0,
    ),
    "wan21_1_3b": DiffusersT2VBackend(
        "wan21_1_3b",
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        fps=16,
        frame_rule="4k+1 or 81",
        width=480,
        height=832,
        steps=40,
        guidance=5.0,
    ),
    "openmontage_shot": OpenMontageBackend(OPENMONTAGE_ROOT),
    "longlive_1_3b": LongLiveBackend(
        repo=LONGLIVE_ROOT,
        models_root=MODEL_ROOT,
        python=PROJECT_ROOT / "bench" / ".venv" / "bin" / "python",
    ),
}


def get_backend(name: str, config: BenchConfig) -> Backend:
    if name == "ours":
        return PipelineBackend(config)
    if name in REGISTRY:
        return REGISTRY[name]
    raise SystemExit(f"unknown backend {name!r}; choose from {', '.join(['ours', *REGISTRY])}")


def describe_backends(config: BenchConfig) -> list[dict]:
    """Availability of every backend, for the run's header and the report."""
    rows = []
    for name in ["ours", *REGISTRY]:
        backend = get_backend(name, config)
        ok, detail = backend.available()
        rows.append({"name": name, "available": ok, "detail": detail, "modality": backend.modality})
    return rows
