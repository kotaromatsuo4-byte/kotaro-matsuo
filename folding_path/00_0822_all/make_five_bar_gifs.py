"""Render every successful five-bar first-hit trajectory in the x-z plane."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageSequence


BASE_DIRECTORY = Path(__file__).resolve().parent
FIRST_HIT_DIRECTORY = BASE_DIRECTORY / "result" / "first_hit"
MANIFEST_PATH = FIRST_HIT_DIRECTORY / "manifest.csv"
MEMBER_PATH = BASE_DIRECTORY / "model" / "five_bar_linkage" / "member.dat"
STEP_PATTERN = re.compile(r"step(\d+)\.dat")

# A common, fixed view makes all D2/D3 paths directly comparable.
X_LIMITS = (-2.2, 2.2)
Z_LIMITS = (-1.0, 2.0)
IMAGE_SIZE = (800, 560)
MARGIN = 30
SUPERSAMPLING = 2
FRAME_DURATION_MS = 50
FIRST_FRAME_DURATION_MS = 600
FINAL_FRAME_DURATION_MS = 1_200


def manifest_rows() -> list[dict[str, str]]:
    with MANIFEST_PATH.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    return [
        row
        for row in rows
        if row["task"] in {"five_d2", "five_d3"}
        and int(row["success"]) == 1
    ]


def trajectory_files(directory: Path) -> list[Path]:
    numbered: list[tuple[int, Path]] = []
    for path in directory.glob("step*.dat"):
        match = STEP_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected trajectory filename: {path}")
        numbered.append((int(match.group(1)), path))
    numbered.sort(key=lambda item: item[0])
    observed = [number for number, _ in numbered]
    if observed != list(range(len(numbered))):
        raise ValueError(f"non-contiguous trajectory: {directory}")
    if not numbered:
        raise ValueError(f"no trajectory DAT files: {directory}")
    return [path for _, path in numbered]


def project_to_pixels(nodes: np.ndarray) -> list[tuple[float, float]]:
    width, height = IMAGE_SIZE
    scale_factor = SUPERSAMPLING
    width *= scale_factor
    height *= scale_factor
    margin = MARGIN * scale_factor

    x_min, x_max = X_LIMITS
    z_min, z_max = Z_LIMITS
    scale = min(
        (width - 2 * margin) / (x_max - x_min),
        (height - 2 * margin) / (z_max - z_min),
    )
    centre_x = width / 2.0
    centre_z = height / 2.0
    data_centre_x = (x_min + x_max) / 2.0
    data_centre_z = (z_min + z_max) / 2.0
    return [
        (
            centre_x + (float(node[0]) - data_centre_x) * scale,
            centre_z - (float(node[2]) - data_centre_z) * scale,
        )
        for node in nodes
    ]


def render_frame(nodes: np.ndarray, members: np.ndarray) -> Image.Image:
    if nodes.shape != (5, 3) or not np.all(np.isfinite(nodes)):
        raise ValueError(f"invalid five-bar node array: shape={nodes.shape}")
    scale_factor = SUPERSAMPLING
    canvas = Image.new(
        "RGB",
        tuple(dimension * scale_factor for dimension in IMAGE_SIZE),
        "white",
    )
    drawing = ImageDraw.Draw(canvas)
    points = project_to_pixels(nodes)
    for start, end in members:
        drawing.line(
            (points[int(start)], points[int(end)]),
            fill="black",
            width=3 * scale_factor,
        )
    radius = 7 * scale_factor
    for x_pixel, z_pixel in points:
        drawing.ellipse(
            (
                x_pixel - radius,
                z_pixel - radius,
                x_pixel + radius,
                z_pixel + radius,
            ),
            fill="white",
            outline="black",
            width=2 * scale_factor,
        )
    canvas = canvas.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    return canvas.convert("P", palette=Image.Palette.ADAPTIVE, colors=16)


def frame_durations(frame_count: int) -> list[int]:
    if frame_count == 1:
        return [FINAL_FRAME_DURATION_MS]
    durations = [FRAME_DURATION_MS] * frame_count
    durations[0] = FIRST_FRAME_DURATION_MS
    durations[-1] = FINAL_FRAME_DURATION_MS
    return durations


def coalesce_identical_frames(
    frames: list[Image.Image], durations: list[int]
) -> tuple[list[Image.Image], list[int]]:
    """Merge visual no-op steps while retaining their elapsed playback time."""
    encoded_frames = [frames[0]]
    encoded_durations = [durations[0]]
    for frame, duration in zip(frames[1:], durations[1:]):
        previous = encoded_frames[-1]
        if (
            frame.tobytes() == previous.tobytes()
            and frame.getpalette() == previous.getpalette()
        ):
            encoded_durations[-1] += duration
        else:
            encoded_frames.append(frame)
            encoded_durations.append(duration)
    return encoded_frames, encoded_durations


def render_trajectory(
    directory: Path, members: np.ndarray
) -> tuple[int, int, Path]:
    paths = trajectory_files(directory)
    frames = [
        render_frame(np.loadtxt(path, dtype=float), members)
        for path in paths
    ]
    durations = frame_durations(len(frames))
    encoded_frames, encoded_durations = coalesce_identical_frames(
        frames, durations
    )
    output_path = directory / "trajectory_xz.gif"
    temporary_path = directory / "trajectory_xz.gif.tmp"
    encoded_frames[0].save(
        temporary_path,
        format="GIF",
        save_all=True,
        append_images=encoded_frames[1:],
        duration=encoded_durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    temporary_path.replace(output_path)
    with Image.open(output_path) as animation:
        if animation.n_frames != len(encoded_frames):
            raise ValueError(
                f"encoded frame count mismatch for {output_path}: "
                f"{animation.n_frames} != {len(encoded_frames)}"
            )
        if animation.size != IMAGE_SIZE:
            raise ValueError(
                f"encoded image size mismatch for {output_path}: "
                f"{animation.size} != {IMAGE_SIZE}"
            )
        encoded_duration = sum(
            int(frame.info.get("duration", 0))
            for frame in ImageSequence.Iterator(animation)
        )
        if encoded_duration != sum(durations):
            raise ValueError(
                f"encoded duration mismatch for {output_path}: "
                f"{encoded_duration} != {sum(durations)}"
            )
    return len(frames), len(encoded_frames), output_path


def main() -> None:
    members = np.atleast_2d(np.loadtxt(MEMBER_PATH, delimiter=",", dtype=int))
    if members.shape != (4, 2):
        raise ValueError(f"unexpected five-bar member array: {members.shape}")

    rows = sorted(
        manifest_rows(),
        key=lambda row: (row["task"], row["method"], int(row["seed"])),
    )
    total_steps = 0
    total_encoded_frames = 0
    for row in rows:
        directory = BASE_DIRECTORY / row["destination"]
        step_count, encoded_count, output_path = render_trajectory(
            directory, members
        )
        expected = int(row["trajectory_dat_count"])
        if step_count != expected:
            raise ValueError(
                f"step count mismatch for {directory}: {step_count} != {expected}"
            )
        total_steps += step_count
        total_encoded_frames += encoded_count
        print(
            f"{row['task']} {row['method']} seed{row['seed']}: "
            f"{step_count} steps, {encoded_count} frames -> "
            f"{output_path.relative_to(BASE_DIRECTORY)}",
            flush=True,
        )
    print(
        f"gifs={len(rows)} steps={total_steps} "
        f"encoded_frames={total_encoded_frames}",
        flush=True,
    )


if __name__ == "__main__":
    main()
