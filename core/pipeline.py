import time
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any

from config import (
    SUPPORTED_VIDEO_EXTS,
    SUPPORTED_SUBTITLE_EXTS,
    SUPPORTED_AUDIO_EXTS,
)
from core.audio_extractor import AudioExtractor, audio_extractor
from core.transcriber import WhisperTranscriber, transcriber
from core.cleaner_translator import GeminiSubtitleCleaner, cleaner_translator
from core.subtitle_parser import SubtitleParser, SubtitleItem


class SubtitlePipeline:
    """End-to-end pipeline for generating clean Korean subtitles from local videos or existing subtitle files."""

    def __init__(
        self,
        extractor: Optional[AudioExtractor] = None,
        stt: Optional[WhisperTranscriber] = None,
        cleaner: Optional[GeminiSubtitleCleaner] = None,
    ):
        self.extractor = extractor or audio_extractor
        self.stt = stt or transcriber
        self.cleaner = cleaner or cleaner_translator

    def process_file(
        self,
        input_path: str | Path,
        existing_subtitle_path: Optional[str | Path] = None,
        output_srt_path: Optional[str | Path] = None,
        source_language: Optional[str] = None,
        custom_instructions: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Path:
        """
        Processes a single video or subtitle file and generates a clean Korean .srt file.

        Args:
            input_path: Path to the video, audio, or subtitle file.
            existing_subtitle_path: Optional path to an existing foreign/dirty subtitle file.
            output_srt_path: Optional custom output path for the final .ko.srt.
            source_language: Optional hint for audio language (e.g. 'en', 'ja', 'zh').
            custom_instructions: Optional instructions for Gemini (character tone, glossary).
            progress_callback: Optional callback(message, progress_percent_0_to_100).

        Returns:
            Path to the generated clean .srt file.
        """
        input_file = Path(input_path).resolve()
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        suffix = input_file.suffix.lower()

        # Determine default output path if not specified
        if output_srt_path is None:
            # e.g., "video.mp4" -> "video.ko.srt"
            if suffix in SUPPORTED_SUBTITLE_EXTS and not input_file.name.endswith(".ko.srt"):
                output_srt_path = input_file.with_name(f"{input_file.stem}.ko.srt")
            else:
                output_srt_path = input_file.with_name(f"{input_file.stem}.ko.srt")
        else:
            output_srt_path = Path(output_srt_path).resolve()

        def notify(msg: str, pct: float):
            if progress_callback:
                progress_callback(msg, pct)
            print(f"[{pct:5.1f}%] {msg}")

        # Case A: Input is directly a Subtitle file
        if suffix in SUPPORTED_SUBTITLE_EXTS:
            notify(f"📄 기존 자막 파일 로드 중: {input_file.name}", 10.0)
            items = SubtitleParser.load(input_file)
            
            def sub_callback(current, total, msg):
                pct = 20.0 + (current / max(1, total)) * 70.0
                notify(f"✨ {msg}", pct)

            items = self.cleaner.process_subtitles(
                items=items,
                custom_instructions=custom_instructions,
                progress_callback=sub_callback,
            )
            notify(f"💾 최종 한국어 자막 저장 중: {output_srt_path.name}", 92.0)
            out_file = SubtitleParser.save_srt(items, output_srt_path, use_translated=True)
            notify(f"🎉 자막 생성 완료: {out_file}", 100.0)
            return out_file

        # Case B: Input is a Video or Audio file with an explicit existing subtitle file
        if existing_subtitle_path and Path(existing_subtitle_path).exists():
            sub_file = Path(existing_subtitle_path).resolve()
            notify(f"📄 지정된 자막 파일 로드: {sub_file.name}", 10.0)
            items = SubtitleParser.load(sub_file)

            def sub_callback(current, total, msg):
                pct = 20.0 + (current / max(1, total)) * 70.0
                notify(f"✨ {msg}", pct)

            items = self.cleaner.process_subtitles(
                items=items,
                custom_instructions=custom_instructions,
                progress_callback=sub_callback,
            )
            notify(f"💾 최종 한국어 자막 저장 중: {output_srt_path.name}", 92.0)
            out_file = SubtitleParser.save_srt(items, output_srt_path, use_translated=True)
            notify(f"🎉 자막 생성 완료: {out_file}", 100.0)
            return out_file

        # Check if an existing same-named subtitle exists in the same folder
        auto_found_sub = self._find_existing_subtitle(input_file)
        if auto_found_sub:
            notify(f"🔍 동일 폴더 내 기존 자막 자동 감지: {auto_found_sub.name}", 10.0)
            items = SubtitleParser.load(auto_found_sub)

            def sub_callback(current, total, msg):
                pct = 20.0 + (current / max(1, total)) * 70.0
                notify(f"✨ {msg}", pct)

            items = self.cleaner.process_subtitles(
                items=items,
                custom_instructions=custom_instructions,
                progress_callback=sub_callback,
            )
            notify(f"💾 최종 한국어 자막 저장 중: {output_srt_path.name}", 92.0)
            out_file = SubtitleParser.save_srt(items, output_srt_path, use_translated=True)
            notify(f"🎉 자막 생성 완료: {out_file}", 100.0)
            return out_file

        # Case C: Video / Audio without subtitle -> Full STT + Translation Pipeline
        temp_wav_path = None
        try:
            # 1. Audio Extraction
            notify(f"🎵 동영상에서 오디오 추출 중 (FFmpeg): {input_file.name}", 10.0)
            temp_wav_path = self.extractor.extract_audio(input_file)

            # 2. Whisper STT (RTX 3090)
            notify("🎙️ 로컬 RTX 3090 음성 인식(STT) 준비 중...", 30.0)
            
            def stt_callback(idx, current_sec, total_sec, msg):
                ratio = min(1.0, current_sec / max(0.1, total_sec))
                pct = 30.0 + ratio * 35.0
                notify(f"🎙️ {msg}", pct)

            items = self.stt.transcribe(
                audio_path=temp_wav_path,
                language=source_language,
                progress_callback=stt_callback,
            )
            notify(f"✅ 음성 인식 완료: 총 {len(items)}개 자막 구간 추출됨", 65.0)

            # 3. Gemini Translation & Grammar/Spacing Cleaning
            def sub_callback(current, total, msg):
                pct = 65.0 + (current / max(1, total)) * 30.0
                notify(f"✨ {msg}", pct)

            items = self.cleaner.process_subtitles(
                items=items,
                custom_instructions=custom_instructions,
                progress_callback=sub_callback,
            )

            # 4. Save Final Subtitle
            notify(f"💾 최종 한국어 자막 저장 중: {output_srt_path.name}", 95.0)
            out_file = SubtitleParser.save_srt(items, output_srt_path, use_translated=True)
            notify(f"🎉 자막 생성 완료: {out_file}", 100.0)
            return out_file

        finally:
            # Cleanup temporary wav file
            if temp_wav_path and Path(temp_wav_path).exists():
                try:
                    Path(temp_wav_path).unlink()
                except Exception:
                    pass

    def process_folder(
        self,
        folder_path: str | Path,
        recursive: bool = False,
        source_language: Optional[str] = None,
        custom_instructions: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[Path]:
        """
        Batch processes all supported video files in a folder.
        """
        folder = Path(folder_path).resolve()
        if not folder.exists() or not folder.is_dir():
            raise NotADirectoryError(f"Directory not found: {folder}")

        files: List[Path] = []
        pattern = "**/*" if recursive else "*"
        for p in folder.glob(pattern):
            if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTS:
                files.append(p)

        total_files = len(files)
        print(f"\n📁 폴더 내 처리 대상 영상 파일: 총 {total_files}개 발견")
        generated_files: List[Path] = []

        for idx, file_path in enumerate(files, start=1):
            print(f"\n==================================================")
            print(f"🎬 [{idx}/{total_files}] 처리 중: {file_path.name}")
            print(f"==================================================")

            if progress_callback:
                progress_callback(idx, total_files, f"처리 중: {file_path.name}")

            try:
                out_srt = self.process_file(
                    input_path=file_path,
                    source_language=source_language,
                    custom_instructions=custom_instructions,
                )
                generated_files.append(out_srt)
            except Exception as e:
                print(f"❌ 오류 발생 ({file_path.name}): {e}")

        return generated_files

    @staticmethod
    def _find_existing_subtitle(video_path: Path) -> Optional[Path]:
        """Searches for existing subtitle file matching the video name in the same directory."""
        parent = video_path.parent
        stem = video_path.stem
        # Candidates to look for: video.srt, video.en.srt, video.smi, video.SMI, etc.
        for ext in SUPPORTED_SUBTITLE_EXTS:
            for cand in [parent / f"{stem}{ext}", parent / f"{stem}{ext.upper()}"]:
                if cand.exists() and not cand.name.endswith(".ko.srt"):
                    return cand
            # Common language suffixes: .en, .ja, .zh, .und, .eng, .jpn, .kor
            for lang in ["en", "ja", "zh", "und", "eng", "jpn", "kor"]:
                for cand_lang in [parent / f"{stem}.{lang}{ext}", parent / f"{stem}.{lang}{ext.upper()}"]:
                    if cand_lang.exists() and not cand_lang.name.endswith(".ko.srt"):
                        return cand_lang
        return None


# Quick singleton instance
pipeline = SubtitlePipeline()
