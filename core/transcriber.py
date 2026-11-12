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

        # Run transcription with faster-whisper on GPU with word-level timestamps
        segments, info = model.transcribe(
            audio_file,
            language=language,
            beam_size=5,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=vad_filter,
            vad_parameters=dict(min_silence_duration_ms=500),
            word_timestamps=True,
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

        # Collect raw segments for generator iteration and progress reporting
        raw_segments = []
        last_reported_time = time.time()
        for idx, seg in enumerate(segments, start=1):
            raw_segments.append(seg)
            if progress_callback and (time.time() - last_reported_time > 0.5 or idx == 1):
                last_reported_time = time.time()
                progress_callback(
                    idx,
                    seg.end,
                    duration,
                    f"음성 인식(STT) 진행 중: {seg.end/60:.1f}분 / {duration/60:.1f}분 ({idx}개 청크 처리됨)"
                )

        # Intelligently split into natural sentence-level subtitle items
        items = self._segment_words_to_subtitles(raw_segments, language=detected_lang)

        if progress_callback:
            progress_callback(
                len(items),
                duration,
                duration,
                f"✅ 음성 인식 완료: 총 {len(items)}개 정밀 자막 구간 생성 완료"
            )

        return items

    def _segment_words_to_subtitles(
        self,
        segments,
        language: Optional[str] = None,
        max_duration: float = 7.0,
        min_duration: float = 0.8,
        max_chars: int = 40,
    ) -> List[SubtitleItem]:
        """
        Groups word-level timestamps into complete, natural sentences and readable subtitle lines.
        Ensures each acoustic segment is treated independently to prevent multi-speaker merging
        and guarantees words/morphemes are never sliced in half.
        """
        is_cjk = language in ("ja", "zh", "yue")
        PUNCT_END = (".", "?", "!", "…", "。", "？", "！")
        PUNCT_CLAUSE = (",", "、", ";", "；")
        HONORIFICS = {"dr.", "mr.", "mrs.", "ms.", "prof.", "st.", "jr.", "sr."}

        def join_words(words_list) -> str:
            if is_cjk:
                return "".join(item.word.strip() for item in words_list)
            return " ".join(item.word.strip() for item in words_list)

        subtitles: List[SubtitleItem] = []
        idx = 1

        for seg in segments:
            words = getattr(seg, "words", None)
            seg_text = seg.text.strip()
            if not seg_text:
                continue

            # If segment has no word timestamps or is already within normal duration and character length
            seg_duration = seg.end - seg.start
            if not words or (seg_duration <= max_duration and len(seg_text) <= max_chars):
                subtitles.append(
                    SubtitleItem(
                        index=idx,
                        start_ms=int(seg.start * 1000),
                        end_ms=int(seg.end * 1000),
                        text=seg_text,
                    )
                )
                idx += 1
                continue

            # For long segments, split strictly at natural sentence/clause/pause boundaries
            curr_words = []
            curr_start = None

            for i, w in enumerate(words):
                w_str = w.word.strip()
                if not w_str:
                    continue

                if curr_start is None:
                    curr_start = w.start

                curr_words.append(w)
                dur = w.end - curr_start
                curr_txt = join_words(curr_words)

                is_honorific = w_str.lower() in HONORIFICS
                ends_sentence = (not is_honorific) and w_str.endswith(PUNCT_END)
                ends_clause = w_str.endswith(PUNCT_CLAUSE)
                pause_after = (words[i + 1].start - w.end) if i + 1 < len(words) else 0.0

                should_split = False
                # 1. Complete sentence with reasonable duration
                if ends_sentence and dur >= min_duration:
                    should_split = True
                # 2. Natural clause boundary (comma) when duration is already moderate (>= 3.5s)
                elif ends_clause and dur >= 3.5:
                    should_split = True
                # 3. Noticeable pause (>= 0.4s) between words with moderate duration
                elif pause_after >= 0.4 and dur >= 3.0:
                    should_split = True
                # 4. Long duration or long text: split at clause/pause or when exceeding max_duration
                elif (dur >= max_duration or len(curr_txt) >= max_chars) and (ends_clause or pause_after >= 0.25 or dur >= 8.5):
                    should_split = True

                if should_split and i + 1 < len(words):
                    sub_text = join_words(curr_words)
                    if sub_text:
                        subtitles.append(
                            SubtitleItem(
                                index=idx,
                                start_ms=int(curr_start * 1000),
                                end_ms=int(w.end * 1000),
                                text=sub_text,
                            )
                        )
                        idx += 1
                    curr_words = []
                    curr_start = None

            # Flush remaining words in this segment
            if curr_words:
                sub_text = join_words(curr_words)
                if sub_text:
                    subtitles.append(
                        SubtitleItem(
                            index=idx,
                            start_ms=int(curr_start * 1000),
                            end_ms=int(curr_words[-1].end * 1000),
                            text=sub_text,
                        )
                    )
                    idx += 1

        return subtitles


# Quick singleton instance
transcriber = WhisperTranscriber()
