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

from faster_whisper import WhisperModel, decode_audio

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

    # Language-specific punctuated prompts prevent Whisper large-v3 from slipping into
    # lowercase/no-punctuation mode when condition_on_previous_text=False.
    PUNCTUATION_PROMPTS = {
        "en": "Hello! Wait a second. What happened? Really? Yes, I understand.",
        "ja": "こんにちは。本当ですか？はい、分かりました！ちょっと待ってください。",
        "ko": "안녕하세요. 정말인가요? 네, 알겠습니다! 잠깐만요.",
        "zh": "你好。真的吗？是的，我明白了！请等一下。",
        "fr": "Bonjour ! Attendez une seconde. Que s'est-il passé ? Vraiment ? Oui, je comprends.",
        "de": "Hallo! Warte mal. Was ist passiert? Wirklich? Ja, ich verstehe.",
        "es": "¡Hola! Espera un segundo. ¿Qué pasó? ¿De verdad? Sí, entiendo.",
    }

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

        # Decode once to numpy array so we can fast-detect language before applying punctuation prompt
        audio_array = decode_audio(audio_file, sampling_rate=16000)

        target_language = language
        if not target_language:
            try:
                detected_code, _, _ = model.detect_language(
                    audio_array,
                    vad_filter=vad_filter,
                    language_detection_segments=1,
                )
                target_language = detected_code
            except Exception:
                target_language = None

        initial_prompt = self.PUNCTUATION_PROMPTS.get(target_language) if target_language else None

        # Run transcription with faster-whisper on GPU with word-level timestamps & punctuation prompt
        # condition_on_previous_text=True keeps proper punctuation/casing across all 30s windows,
        # while vad_filter + hallucination_silence_threshold + temperature fallback prevent repetition loops.
        segments, info = model.transcribe(
            audio_array,
            language=target_language,
            beam_size=5,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            condition_on_previous_text=True,
            initial_prompt=initial_prompt,
            vad_filter=vad_filter,
            vad_parameters=dict(min_silence_duration_ms=350, speech_pad_ms=150),
            word_timestamps=True,
            hallucination_silence_threshold=2.0,
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
        Includes:
        - Independent acoustic segment handling (prevents multi-speaker merging)
        - Word-level stretched timestamp sanitization (prevents VAD/DTW silence-spanning drift)
        - Inter-word silence-jump detection (snaps orphaned leading fragments or splits across long silences)
        """
        is_cjk = language in ("ja", "zh", "yue")
        PUNCT_END = (".", "?", "!", "…", "。", "？", "！")
        PUNCT_CLAUSE = (",", "、", ";", "；")
        HONORIFICS = {"dr.", "mr.", "mrs.", "ms.", "prof.", "st.", "jr.", "sr."}
        MAX_WORD_DUR = 1.8
        SILENCE_JUMP_SEC = 1.8
        SHORT_FRAG_LIMIT = 6 if is_cjk else 3

        def join_words(words_list) -> str:
            if is_cjk:
                return "".join(w_str for _, _, w_str in words_list)
            return " ".join(w_str for _, _, w_str in words_list)

        def sanitize_words(raw_words) -> tuple[list[tuple[float, float, str]], bool]:
            """
            Sanitizes abnormally stretched single-word durations caused by VAD chunk splicing
            or DTW alignment across long silences/music.
            Returns (sanitized_words, had_anomaly).
            """
            cleaned = []
            for w in raw_words:
                w_str = w.word.strip()
                if not w_str:
                    continue
                cleaned.append([float(w.start), float(w.end), w_str])

            had_anomaly = False
            n = len(cleaned)
            for i in range(n):
                st, en, _ = cleaned[i]
                dur = en - st
                if dur > MAX_WORD_DUR:
                    had_anomaly = True
                    gap_before = (st - cleaned[i - 1][1]) if i > 0 else 999.0
                    gap_after = (cleaned[i + 1][0] - en) if i + 1 < n else 999.0

                    if i + 1 < n and gap_after < 0.5:
                        # Anchored to following words: snap start forward close to end
                        cleaned[i][0] = max(st, en - 0.5)
                    elif i > 0 and gap_before < 0.5:
                        # Anchored to preceding words: snap end backward close to start
                        cleaned[i][1] = min(en, st + 0.6)
                    else:
                        # Isolated stretched word: anchor near end (actual speech onset)
                        cleaned[i][0] = max(st, en - 0.8)

            # Also check if there is any large inter-word silence jump
            for i in range(n - 1):
                if cleaned[i + 1][0] - cleaned[i][1] >= SILENCE_JUMP_SEC:
                    had_anomaly = True
                    break

            return [(st, en, txt) for st, en, txt in cleaned], had_anomaly

        subtitles: List[SubtitleItem] = []
        idx = 1

        def append_item(start_s: float, end_s: float, text: str):
            nonlocal idx
            text = text.strip()
            if not text:
                return
            start_ms = max(0, int(round(start_s * 1000)))
            end_ms = max(start_ms + 200, int(round(end_s * 1000)))
            if subtitles and start_ms < subtitles[-1].end_ms:
                # Prevent timestamp overlap with previous subtitle
                start_ms = subtitles[-1].end_ms
                if end_ms < start_ms + 200:
                    end_ms = start_ms + 200
            subtitles.append(
                SubtitleItem(
                    index=idx,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    text=text,
                )
            )
            idx += 1

        for seg in segments:
            raw_words = getattr(seg, "words", None)
            seg_text = seg.text.strip()
            if not seg_text:
                continue

            if not raw_words:
                append_item(seg.start, seg.end, seg_text)
                continue

            words, had_anomaly = sanitize_words(raw_words)
            if not words:
                append_item(seg.start, seg.end, seg_text)
                continue

            # Fast path ONLY if segment is short, has zero word-timestamp anomalies/jumps,
            # AND contains no internal sentence boundary or conversational pause (>= 0.55s).
            seg_start = words[0][0]
            seg_end = words[-1][1]
            seg_duration = seg_end - seg_start
            has_internal_break = False
            for i in range(len(words) - 1):
                w_txt = words[i][2]
                if (w_txt.lower() not in HONORIFICS) and w_txt.endswith(PUNCT_END):
                    has_internal_break = True
                    break
                if words[i + 1][0] - words[i][1] >= 0.55:
                    has_internal_break = True
                    break

            if (
                not had_anomaly
                and not has_internal_break
                and seg_duration <= max_duration
                and len(seg_text) <= max_chars
            ):
                append_item(seg_start, seg_end, seg_text)
                continue

            # Split strictly at natural sentence/clause/pause boundaries while handling silence jumps
            STRONG_TURN_END = ("?", "!", "？", "！")
            curr_words: list[tuple[float, float, str]] = []
            curr_start: Optional[float] = None

            for i, (w_start, w_end, w_str) in enumerate(words):
                if curr_start is None:
                    curr_start = w_start

                curr_words.append((w_start, w_end, w_str))
                dur = w_end - curr_start
                curr_txt = join_words(curr_words)

                is_honorific = w_str.lower() in HONORIFICS
                ends_sentence = (not is_honorific) and w_str.endswith(PUNCT_END)
                ends_strong_turn = w_str.endswith(STRONG_TURN_END)
                ends_clause = w_str.endswith(PUNCT_CLAUSE)
                pause_after = (words[i + 1][0] - w_end) if i + 1 < len(words) else 0.0

                # Handle large silence jump (>= 1.8s) caused by VAD chunk splicing or long pauses
                if pause_after >= SILENCE_JUMP_SEC and i + 1 < len(words):
                    next_start = words[i + 1][0]
                    next_word = words[i + 1][2]
                    is_lowercase_continuation = is_cjk or (bool(next_word) and next_word[0].islower())
                    is_huge_vad_splice = (
                        pause_after >= 5.0
                        and not ends_sentence
                        and len(curr_words) <= (4 if is_cjk else 2)
                    )
                    is_mid_sentence_vad_splice = (
                        pause_after >= 2.5
                        and not ends_sentence
                        and len(curr_words) <= SHORT_FRAG_LIMIT
                        and is_lowercase_continuation
                    )
                    if is_huge_vad_splice or is_mid_sentence_vad_splice:
                        # Orphaned leading fragment snapped onto an earlier noise before a long gap:
                        # Move start forward to right before the real speech continuation.
                        est_lead = min(1.5, 0.35 * len(curr_words))
                        curr_start = max(curr_start, next_start - est_lead)
                        continue
                    else:
                        # Genuine separate utterance before a long silence: force split cleanly
                        append_item(curr_start, w_end, curr_txt)
                        curr_words = []
                        curr_start = None
                        continue

                should_split = False
                next_word_str = words[i + 1][2] if i + 1 < len(words) else ""
                next_starts_sentence = is_cjk or (bool(next_word_str) and next_word_str[0].isupper())
                # 1. Strong turn ending (? / !) followed by another utterance -> split cleanly
                if ends_strong_turn and (pause_after >= 0.15 or dur >= 0.6):
                    should_split = True
                # 2. Complete sentence (. / 。 / …) with reasonable duration or speaker pause
                elif ends_sentence and (dur >= min_duration or pause_after >= 0.25):
                    should_split = True
                # 3. Clear speaker/breath pause (>= 0.65s) without punctuation:
                #    Only split if the next word starts a new sentence/turn (uppercase) OR phrase is substantial,
                #    avoiding dramatic mid-phrase pauses after 1-2 words (e.g. "I... can't", "face... down").
                elif pause_after >= 0.65 and (
                    (next_starts_sentence and len(curr_words) >= (2 if is_cjk else 1))
                    or (len(curr_words) >= (4 if is_cjk else 3) and dur >= 1.5)
                ):
                    should_split = True
                # 4. Natural clause boundary (comma) when duration is already moderate (>= 3.5s)
                elif ends_clause and dur >= 3.5:
                    should_split = True
                # 5. Moderate pause (>= 0.4s) when duration is already moderate (>= 3.0s)
                elif pause_after >= 0.4 and dur >= 3.0:
                    should_split = True
                # 6. Long duration or long text: split at clause/pause or when exceeding max_duration
                elif (dur >= max_duration or len(curr_txt) >= max_chars) and (
                    ends_clause or pause_after >= 0.25 or dur >= 8.5
                ):
                    should_split = True

                if should_split and i + 1 < len(words):
                    append_item(curr_start, w_end, curr_txt)
                    curr_words = []
                    curr_start = None

            # Flush remaining words in this segment
            if curr_words and curr_start is not None:
                append_item(curr_start, curr_words[-1][1], join_words(curr_words))

        return subtitles


# Quick singleton instance
transcriber = WhisperTranscriber()
