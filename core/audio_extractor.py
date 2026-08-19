import subprocess
import shutil
from pathlib import Path
from typing import Optional, Callable
from config import TEMP_DIR, SUPPORTED_AUDIO_EXTS


class AudioExtractor:
    """Extracts clean audio tracks from video files using FFmpeg."""

    def __init__(self, ffmpeg_bin: str = "ffmpeg"):
        self.ffmpeg_bin = ffmpeg_bin

    def extract_audio(
        self,
        media_path: str | Path,
        output_wav_path: Optional[str | Path] = None,
        sample_rate: int = 16000,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Path:
        """
        Extracts audio from a video/audio file into a 16kHz mono WAV format (ideal for Whisper STT).

        Args:
            media_path: Path to the input video or audio file.
            output_wav_path: Destination path for the extracted .wav file.
            sample_rate: Audio sample rate in Hz (default: 16000 for Whisper).
            progress_callback: Optional callback for status messages.

        Returns:
            Path object pointing to the output .wav file.
        """
        input_path = Path(media_path).resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input media file not found: {input_path}")

        # If the input is already a 16kHz mono WAV, we can check or just re-export cleanly
        if output_wav_path is None:
            output_wav_path = TEMP_DIR / f"{input_path.stem}_temp_audio.wav"
        else:
            output_wav_path = Path(output_wav_path).resolve()

        output_wav_path.parent.mkdir(parents=True, exist_ok=True)

        if progress_callback:
            progress_callback(f"오디오 추출 시작: {input_path.name}")

        # FFmpeg command: extract audio to 16kHz, mono, 16-bit PCM WAV
        cmd = [
            self.ffmpeg_bin,
            "-y",  # Overwrite output file if exists
            "-i", str(input_path),
            "-vn",  # Disable video
            "-acodec", "pcm_s16le",  # Uncompressed 16-bit PCM
            "-ar", str(sample_rate),  # Sample rate (16kHz)
            "-ac", "1",  # Mono channel
            "-threads", "4",
            str(output_wav_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"FFmpeg audio extraction failed for {input_path.name}:\n{e.stderr}"
            ) from e

        if progress_callback:
            progress_callback(f"오디오 추출 완료: {output_wav_path.name}")

        return output_wav_path


# Quick singleton instance
audio_extractor = AudioExtractor()
