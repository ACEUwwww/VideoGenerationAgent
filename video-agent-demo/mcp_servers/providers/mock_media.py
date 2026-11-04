"""Mock providers that still produce real, playable media.

The point is to keep the whole pipeline honest before the vendor API arrives:
images are real PNGs, videos are real H.264 1080x1920 clips animated from those
images, and audio is a real (silent) AAC track of the shot's duration. Only the
creative content is synthetic, so the composer, subtitle burn-in and the
frontend preview are exercised for real.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from mcp_servers.providers.base import ImageRequest, SpeechRequest, VideoRequest
from mcp_servers.providers.ffmpeg_utils import NO_WINDOW, ffmpeg_bin, font_path, run

PALETTE = (
    ((14, 18, 32), (58, 40, 84)),
    ((30, 12, 20), (110, 32, 48)),
    ((10, 26, 28), (24, 84, 76)),
    ((32, 22, 10), (128, 84, 30)),
    ((18, 18, 24), (60, 60, 84)),
)


def _seed(prompt: str) -> int:
    return int(hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8], 16)


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(font_path()), size)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    """Wrap on word boundaries so an English prompt does not split mid-word."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) > width and current:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


class MockImageProvider:
    name = "mock"

    def generate(self, request: ImageRequest) -> dict:
        width, height = request.width, request.height
        seed = request.seed or _seed(request.prompt)
        top, bottom = PALETTE[seed % len(PALETTE)]

        image = Image.new("RGB", (width, height), top)
        draw = ImageDraw.Draw(image)
        for y in range(height):
            ratio = y / max(height - 1, 1)
            color = tuple(int(top[c] + (bottom[c] - top[c]) * ratio) for c in range(3))
            draw.line([(0, y), (width, y)], fill=color)

        # Vignette bands act as a stand-in for cinematic framing.
        for offset in range(0, 420, 6):
            alpha = int(140 * (1 - offset / 420))
            draw.rectangle([0, offset, width, offset + 6], fill=(0, 0, 0) + (alpha,), width=0)

        margin = 72
        badge_font = _font(34)
        title_font = _font(58)
        body_font = _font(44)
        meta_font = _font(30)

        draw.text((margin, 300), f"{request.shot_id.upper()}", font=badge_font, fill=(255, 255, 255))
        draw.line([(margin, 360), (margin + 180, 360)], fill=(255, 255, 255), width=3)

        y = 420
        for line in _wrap(draw, request.caption or request.prompt, title_font, width - 2 * margin)[:6]:
            draw.text((margin, y), line, font=title_font, fill=(245, 245, 250))
            y += 78

        y += 40
        for line in _wrap(draw, request.prompt, body_font, width - 2 * margin)[:12]:
            draw.text((margin, y), line, font=body_font, fill=(205, 205, 215))
            y += 58

        draw.text(
            (margin, height - 210),
            f"MOCK FRAME · {width}x{height} · seed {seed}",
            font=meta_font,
            fill=(160, 160, 175),
        )
        draw.text((margin, height - 150), "replace IMAGE_PROVIDER to render real art", font=meta_font, fill=(160, 160, 175))

        target = request.project_dir / "images" / f"{request.shot_id}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, "PNG")
        return {
            "image_path": str(target.relative_to(request.project_dir)),
            "provider": self.name,
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "seed": seed,
            "width": width,
            "height": height,
        }


class MockVideoProvider:
    name = "mock"

    def generate(self, request: VideoRequest) -> dict:
        target = request.project_dir / "videos" / f"{request.shot_id}.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)
        frames = max(int(round(request.duration_s * request.fps)), request.fps)
        seed = request.seed or _seed(request.motion_prompt)

        # Push-in on the keyframe so the clip carries real motion, not a freeze.
        direction = "in" if seed % 2 == 0 else "out"
        if direction == "in":
            zoom = "min(1.0+0.0016*on,1.16)"
        else:
            zoom = "max(1.16-0.0016*on,1.0)"
        motion = "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        pan_y = "y='ih/2-(ih/zoom/2)-8*sin(on/24)'" if direction == "in" else "y='ih/2-(ih/zoom/2)'"

        vf = (
            f"scale={request.width * 2}:{request.height * 2},"
            f"zoompan=z='{zoom}':{motion.replace(pan_y, pan_y)}:d={frames}:s={request.width}x{request.height}:fps={request.fps},"
            f"format=yuv420p"
        )
        run(
            [
                ffmpeg_bin(),
                "-y",
                "-loop",
                "1",
                "-i",
                str(request.image_path),
                "-t",
                f"{request.duration_s:.3f}",
                "-vf",
                vf,
                "-r",
                str(request.fps),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(target),
            ]
        )
        return {
            "video_path": str(target.relative_to(request.project_dir)),
            "provider": self.name,
            "motion_prompt": request.motion_prompt,
            "duration_s": request.duration_s,
            "frames": frames,
            "direction": direction,
        }


class MockSpeechProvider:
    name = "mock"

    def synthesize(self, request: SpeechRequest) -> dict:
        target = request.project_dir / "audio" / f"{request.shot_id}.m4a"
        target.parent.mkdir(parents=True, exist_ok=True)
        duration = max(float(request.duration_s), 0.5)
        proc = subprocess.run(
            [
                ffmpeg_bin(),
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"anullsrc=r=44100:cl=mono",
                "-t",
                f"{duration:.3f}",
                "-c:a",
                "aac",
                "-b:a",
                "96k",
                str(target),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            encoding="utf-8",
            errors="replace",
            creationflags=NO_WINDOW,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"silent audio generation failed: {proc.stderr[-600:]}")
        return {
            "audio_path": str(target.relative_to(request.project_dir)),
            "provider": self.name,
            "voice": request.voice or "silent",
            "duration_s": duration,
            "text": request.text,
        }
