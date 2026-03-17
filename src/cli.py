import asyncio
import subprocess
import signal
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import typer
from rich.console import Console

from .job_runner import JobRunner
from .models import JobStatus, JobType

app = typer.Typer(
    name="video-processing-job-service",
    help="CLI for video frame extraction and composition without Docker",
)
console = Console()

DATA_DIR = Path("/app/data") if Path("/app/data").exists() else Path("data")


class CliJobRunner(JobRunner):
    """Job runner with CLI progress output."""

    def __init__(
        self,
        job_ref: Optional[dict[str, Any]],
        get_status: Callable[[], str],
    ):
        super().__init__(job_ref, get_status)
        self._last_progress = -1

    async def run(self) -> dict[str, Any]:
        """Run the job with progress output."""
        input_params = self._job_ref.get("input_params", {}) if self._job_ref else {}
        job_type = input_params.get("job_type", "extract")

        if job_type == JobType.EXTRACT:
            return await self._run_extract(input_params)
        elif job_type == JobType.COMPOSE:
            return await self._run_compose(input_params)
        else:
            raise ValueError(f"Unknown job_type: {job_type}")

    async def _run_extract(self, input_params: dict[str, Any]) -> dict[str, Any]:
        """Run the frame extraction job with progress output."""
        input_file = input_params.get("input_file")
        output_dir = input_params.get("output_dir")
        auto_crop = input_params.get("auto_crop", True)

        if not input_file or not output_dir:
            raise ValueError("input_file and output_dir are required")

        input_path = Path(DATA_DIR) / input_file
        output_path = Path(DATA_DIR) / output_dir

        if not input_path.exists():
            raise ValueError(f"Input file not found: {input_path}")

        output_path.mkdir(parents=True, exist_ok=True)

        from .job_runner import JobRunner

        runner = JobRunner(None, lambda: "running")
        metadata = await runner._extract_metadata(input_path)

        crop_x = None
        crop_y = None
        crop_width = None
        crop_height = None

        if auto_crop and metadata.duration_seconds > 30:
            crop_result = await runner._detect_crop(
                input_path, metadata.duration_seconds
            )
            if crop_result:
                crop_width, crop_height, crop_x, crop_y = crop_result
                metadata.crop_width = crop_width
                metadata.crop_height = crop_height
                metadata.crop_x = crop_x
                metadata.crop_y = crop_y

                rotation = metadata.rotation
                sar = metadata.sample_aspect_ratio

                if rotation in (90, -90, 270, -270):
                    metadata.display_width = crop_height
                    metadata.display_height = round(crop_width * sar)
                else:
                    metadata.display_width = round(crop_width * sar)
                    metadata.display_height = crop_height

                metadata.display_width = (metadata.display_width // 2) * 2
                metadata.display_height = (metadata.display_height // 2) * 2

        runner._save_metadata(output_path, metadata)

        frame_dir = output_path / "frame"
        audio_dir = output_path / "audio"
        subtitle_dir = output_path / "subtitle"
        frame_dir.mkdir(exist_ok=True)
        audio_dir.mkdir(exist_ok=True)
        subtitle_dir.mkdir(exist_ok=True)

        output_pattern = str(frame_dir / "frame_%04d.png")

        console.print(f"[cyan]Starting extraction:[/cyan] {input_file} -> {output_dir}")

        if crop_width and crop_height:
            video_filter = f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y},scale={metadata.display_width}:{metadata.display_height}"
        else:
            video_filter = f"scale={metadata.display_width}:{metadata.display_height}"

        ffmpeg_args = [
            "ffmpeg",
            "-i",
            str(input_path),
            "-y",
            "-map",
            "0:v",
            "-vf",
            video_filter,
            output_pattern,
        ]

        for track in metadata.audio_tracks:
            ffmpeg_args.extend(
                [
                    "-map",
                    f"0:{track.stream_index}",
                    "-c:a",
                    "copy",
                    "-y",
                    str(output_path / track.filename),
                ]
            )

        copy_subtitle_codecs = {"subrip", "srt", "ass", "ssa", "webvtt", "vtt"}
        bitmap_subtitle_codecs = {
            "dvbsub",
            "dvd_subtitle",
            "hdmv_pgs_subtitle",
            "vobsub",
        }
        for track in metadata.subtitle_tracks:
            if track.codec in copy_subtitle_codecs:
                ffmpeg_args.extend(
                    [
                        "-map",
                        f"0:{track.stream_index}",
                        "-c:s",
                        "copy",
                        "-y",
                        str(output_path / track.filename),
                    ]
                )
            elif track.codec in bitmap_subtitle_codecs:
                pass
            else:
                output_filename = track.filename.rsplit(".", 1)[0] + ".srt"
                ffmpeg_args.extend(
                    [
                        "-map",
                        f"0:{track.stream_index}",
                        "-c:s",
                        "srt",
                        "-y",
                        str(output_path / output_filename),
                    ]
                )

        process = await asyncio.create_subprocess_exec(
            *ffmpeg_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self._update_progress(10)
        console.print("[cyan]Progress:[/cyan] 10% - Extracting frames...")

        try:
            await process.wait()
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            console.print("[yellow]Job cancelled[/yellow]")
            return {"cancelled": True}

        if process.returncode != 0:
            if process.stderr is not None:
                stderr = await process.stderr.read()
                error_msg = stderr.decode() if stderr else "Unknown error"
            else:
                error_msg = "Unknown error"
            raise RuntimeError(f"FFmpeg failed: {error_msg}")

        frame_files = sorted(
            output_path.glob("frame/frame_*.png"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        frame_count = len(frame_files)

        self._extract_bitmap_subtitles(
            input_path, output_path, metadata.subtitle_tracks
        )

        self._update_progress(100)
        console.print("[cyan]Progress:[/cyan] 100% - Complete!")

        return {
            "completed": True,
            "job_type": "extract",
            "input_file": input_file,
            "output_dir": output_dir,
            "frame_count": frame_count,
            "metadata_file": f"{output_dir}/metadata.json",
            "audio_track_count": len(metadata.audio_tracks),
            "subtitle_track_count": len(metadata.subtitle_tracks),
        }

    def _extract_bitmap_subtitles(
        self,
        input_path: Path,
        output_path: Path,
        metadata: Any,
    ) -> None:
        """Extract bitmap-based subtitles using mkvextract."""
        bitmap_subtitle_codecs = {
            "dvbsub",
            "dvd_subtitle",
            "hdmv_pgs_subtitle",
            "vobsub",
        }
        subtitle_dir = output_path / "subtitle"
        subtitle_dir.mkdir(exist_ok=True)
        subtitle_tracks = (
            metadata.subtitle_tracks
            if hasattr(metadata, "subtitle_tracks")
            else metadata
        )
        for track in subtitle_tracks:
            if track.codec not in bitmap_subtitle_codecs:
                continue
            try:
                output_file = subtitle_dir / f"subtitle_{track.stream_index}.sub"
                result = subprocess.run(
                    [
                        "mkvextract",
                        "tracks",
                        str(input_path),
                        f"{track.stream_index}:{output_file}",
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    console.print(
                        f"[yellow]Warning:[/yellow] Failed to extract subtitle "
                        f"track {track.stream_index}: {result.stderr}"
                    )
            except FileNotFoundError:
                console.print(
                    "[yellow]Warning:[/yellow] mkvextract not found, "
                    "skipping bitmap subtitle extraction"
                )
                break

    async def _run_compose(self, input_params: dict[str, Any]) -> dict[str, Any]:
        """Run the frame composition job with progress output."""
        input_dir = input_params.get("input_dir")
        output_file = input_params.get("output_file")

        if not input_dir or not output_file:
            raise ValueError("input_dir and output_file are required")

        input_path = Path(DATA_DIR) / input_dir
        output_path = Path(DATA_DIR) / output_file

        if not input_path.exists():
            raise ValueError(f"Input directory not found: {input_path}")

        metadata_path = input_path / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"Metadata file not found: {metadata_path}")

        from .job_runner import JobRunner

        runner = JobRunner(None, lambda: "running")
        metadata = runner._load_metadata(metadata_path)

        frame_files = sorted(
            input_path.glob("frame/frame_*.png"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        if not frame_files:
            raise ValueError(f"No frame files found in: {input_path / 'frame'}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        input_pattern = str(input_path / "frame" / "frame_%04d.png")

        console.print(
            f"[cyan]Starting composition:[/cyan] {input_dir} -> {output_file}"
        )

        ffmpeg_args = [
            "ffmpeg",
            "-framerate",
            str(metadata.fps),
            "-i",
            input_pattern,
            "-y",
        ]

        audio_extensions = [
            "aac",
            "mp3",
            "m4a",
            "ac3",
            "eac3",
            "flac",
            "ogg",
            "opus",
            "wav",
        ]
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(sorted(input_path.glob(f"audio/audio_*.{ext}")))
        audio_files.sort()

        subtitle_extensions = ["srt", "ass", "vtt"]
        subtitle_files = []
        for ext in subtitle_extensions:
            subtitle_files.extend(sorted(input_path.glob(f"subtitle/subtitle_*.{ext}")))
        subtitle_files.sort()

        audio_index = 0
        for audio_file in audio_files:
            ffmpeg_args.extend(["-i", str(audio_file)])

        for subtitle_file in subtitle_files:
            ffmpeg_args.extend(["-i", str(subtitle_file)])

        ffmpeg_args.extend(
            [
                "-map",
                "0:v",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
            ]
        )

        for _ in audio_files:
            ffmpeg_args.extend(["-map", f"{audio_index + 1}:a", "-c:a", "copy"])
            audio_index += 1

        subtitle_input_offset = 1 + len(audio_files)
        for i in range(len(subtitle_files)):
            ffmpeg_args.extend(
                ["-map", f"{subtitle_input_offset + i}:s", "-c:s", "copy"]
            )

        ffmpeg_args.append(str(output_path))

        process = await asyncio.create_subprocess_exec(
            *ffmpeg_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self._update_progress(10)
        console.print("[cyan]Progress:[/cyan] 10% - Composing frames...")

        try:
            await process.wait()
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            console.print("[yellow]Job cancelled[/yellow]")
            return {"cancelled": True}

        if process.returncode != 0:
            if process.stderr is not None:
                stderr = await process.stderr.read()
                error_msg = stderr.decode() if stderr else "Unknown error"
            else:
                error_msg = "Unknown error"
            raise RuntimeError(f"FFmpeg failed: {error_msg}")

        self._update_progress(100)
        console.print("[cyan]Progress:[/cyan] 100% - Complete!")

        return {
            "completed": True,
            "job_type": "compose",
            "input_dir": input_dir,
            "output_file": output_file,
            "frame_count": len(frame_files),
            "fps": metadata.fps,
            "audio_track_count": len(metadata.audio_tracks),
            "subtitle_track_count": len(metadata.subtitle_tracks),
        }


def run_cli_job(
    job_id: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Run a job synchronously for the CLI."""
    job_ref: dict[str, Any] = {
        "id": job_id,
        "status": JobStatus.RUNNING,
        "progress": 0,
        "input_params": params,
    }

    def get_status() -> str:
        return job_ref["status"]

    async def run_with_progress():
        runner = CliJobRunner(job_ref, get_status)
        try:
            result = await runner.run()
            if job_ref["status"] == JobStatus.RUNNING:
                job_ref["status"] = JobStatus.COMPLETED
                job_ref["result"] = result
        except Exception as e:
            job_ref["status"] = JobStatus.FAILED
            job_ref["error"] = str(e)

    original_handler = signal.signal(signal.SIGINT, signal.SIG_DFL)

    def signal_handler(sig, frame):
        job_ref["status"] = JobStatus.CANCELLED
        console.print("\n[yellow]Received interrupt, cancelling job...[/yellow]")
        sys.exit(130)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        asyncio.run(run_with_progress())
    finally:
        signal.signal(signal.SIGINT, original_handler)

    return job_ref


@app.command()
def run(
    job_id: Optional[str] = typer.Option(
        None,
        "--job-id",
        "-j",
        help="Job identifier (auto-generated if not provided)",
    ),
    job_type: JobType = typer.Option(
        ...,
        "--job-type",
        "-t",
        help="Job type: extract or compose",
    ),
    input_file: Optional[str] = typer.Option(
        None,
        "--input",
        "-i",
        help="Input video file path (for extract)",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for frames or output video file",
    ),
    input_dir: Optional[str] = typer.Option(
        None,
        "--input-dir",
        help="Input directory containing frames (for compose)",
    ),
    output_file: Optional[str] = typer.Option(
        None,
        "--output-file",
        help="Output video file path (for compose)",
    ),
    auto_crop: bool = typer.Option(
        True,
        "--auto-crop/--no-auto-crop",
        help="Automatically crop black bars from video (default: enabled)",
    ),
) -> None:
    """
    Run a video processing job (extract or compose).

    Extract: Extract frames from a video to PNG images.
        video-processing-job-service run -t extract -i video.mp4 -o output_frames

    Compose: Create a video from PNG frames.
        video-processing-job-service run -t compose -i output_frames -o composed.mp4
    """
    import uuid

    if job_id is None:
        job_id = str(uuid.uuid4())

    if job_type == JobType.EXTRACT:
        if not input_file or not output_dir:
            console.print("[red]Error:[/red] extract requires --input and --output")
            raise typer.Exit(code=1)
        param_dict = {
            "job_type": job_type.value,
            "input_file": input_file,
            "output_dir": output_dir,
            "auto_crop": auto_crop,
        }
        console.print(f"[bold]Starting extract job:[/bold] {job_id}")
    elif job_type == JobType.COMPOSE:
        if not input_dir or not output_file:
            console.print(
                "[red]Error:[/red] compose requires --input-dir and --output-file"
            )
            raise typer.Exit(code=1)
        param_dict = {
            "job_type": job_type.value,
            "input_dir": input_dir,
            "output_file": output_file,
        }
        console.print(f"[bold]Starting compose job:[/bold] {job_id}")
    else:
        console.print(f"[red]Error:[/red] unknown job type: {job_type}")
        raise typer.Exit(code=1)

    job_result = run_cli_job(job_id, param_dict)

    if job_result["status"] == JobStatus.COMPLETED:
        console.print("[green]Job completed successfully![/green]")
        console.print(f"Result: {job_result.get('result')}")
    elif job_result["status"] == JobStatus.CANCELLED:
        console.print("[yellow]Job was cancelled[/yellow]")
        raise typer.Exit(code=130)
    else:
        console.print(f"[red]Job failed:[/red] {job_result.get('error')}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
