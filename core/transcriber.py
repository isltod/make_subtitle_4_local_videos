import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Callable

# Add torch/lib to Windows DLL search path for CUDA GPU acceleration in ctranslate2
torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
if torch_lib.exists():
    try:
        os.add_dll_directory(str(torch_lib))
    except Exception:
        pass
    os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")

from faster_whisper import WhisperModel

from config import (
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_VAD_FILTER,
)
from core.subtitle_parser import SubtitleItem, SubtitleParser


class WhisperTranscriber:
    """Wraps faster-whisper for speech-to-text transcription on GPU (RTX 3090)."""

    _model_instance: Optional[WhisperModel] = None

    def __init__(
        self,
        model_size: str = WHISPER_MODEL_SIZE,
        device: str = WHISPER_DEVICE,
        compute_type: str = WHISPER_COMPUTE_TYPE,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type

    def _get_model(self) -> WhisperModel:
        """Loads and caches the WhisperModel in GPU VRAM (without Windows symlink issues)."""
        if WhisperTranscriber._model_instance is None:
            # Check if local model folder exists or download cleanly to models/
            local_model_dir = Path(__file__).resolve().parent.parent / "models" / f"faster-whisper-{self.model_size}"
            
            if local_model_dir.exists() and (local_model_dir / "model.bin").exists():
                model_target = str(local_model_dir)
            else:
                try:
                    from huggingface_hub import snapshot_download
                    repo_id = f"Systran/faster-whisper-{self.model_size}"
                    print(f"📦 Downloading/Verifying Whisper model [{repo_id}] into local folder...")
                    model_target = snapshot_download(
                        repo_id=repo_id,
                        local_dir=str(local_model_dir),
                    )
                except Exception:
                    # Fallback to default name
                    model_target = self.model_size

            print(f"📦 Loading Whisper model [{model_target}] on [{self.device}:{self.compute_type}]...")
            t0 = time.time()
            WhisperTranscriber._model_instance = WhisperModel(
                model_target,
                device=self.device,
                compute_type=self.compute_type,
            )
            print(f"✅ Whisper model loaded in {time.time() - t0:.2f}s")
        return WhisperTranscriber._model_instance

    def transcribe(
        self,
        audio_path: str | Path,
        language: Optional[str] = None,
        vad_filter: bool = WHISPER_VAD_FILTER,
        progress_callback: Optional[Callable[[int, float, float, str], None]] = None,
    ) -> List[SubtitleItem]:
        """
        Transcribes audio file to a list of SubtitleItem objects with precise timestamps.

        Args:
            audio_path: Path to the .wav audio file.
            language: Optional language code (e.g. 'en', 'ja', 'ko', 'zh', or None for auto-detection).
            vad_filter: Whether to apply Silero VAD to filter out silence/background noise.
            progress_callback: Optional callback(segment_index, current_timestamp_sec, total_duration_sec, text).

        Returns:
            List of SubtitleItem objects.
        """
        model = self._get_model()
        audio_file = str(Path(audio_path).resolve())

        # Run transcription with faster-whisper on GPU
        segments, info = model.transcribe(
            audio_file,
            language=language,
            beam_size=5,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=vad_filter,
            vad_parameters=dict(min_silence_duration_ms=500),
            word_timestamps=False,
        )

        detected_lang = info.language
        lang_prob = info.language_probability
        duration = info.duration

        if progress_callback:
            progress_callback(
                0,
                0.0,
                duration,
                f"감지된 언어: [{detected_lang.upper()}] (신뢰도: {lang_prob*100:.1f}%), 총 재생시간: {duration:.1f}초"
            )

        items: List[SubtitleItem] = []
        last_reported_time = time.time()
        for idx, seg in enumerate(segments, start=1):
            text = seg.text.strip()
            if not text:
                continue
            item = SubtitleItem(
                index=idx,
                start_ms=int(seg.start * 1000),
                end_ms=int(seg.end * 1000),
                text=text,
            )
            items.append(item)

            if progress_callback and (time.time() - last_reported_time > 0.5 or idx == 1):
                last_reported_time = time.time()
                progress_callback(
                    idx,
                    seg.end,
                    duration,
                    f"음성 인식(STT) 진행 중: {seg.end/60:.1f}분 / {duration/60:.1f}분 ({len(items)}개 구간 추출됨)"
                )

        if progress_callback:
            progress_callback(
                len(items),
                duration,
                duration,
                f"✅ 음성 인식 완료: 총 {len(items)}개 자막 구간 추출 완료"
            )

        return items


# Quick singleton instance
transcriber = WhisperTranscriber()
