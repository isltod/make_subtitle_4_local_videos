from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import pysubs2


@dataclass
class SubtitleItem:
    """Represents a single subtitle entry with index, timing, and text."""
    index: int  # 1-based index
    start_ms: int  # Start time in milliseconds
    end_ms: int  # End time in milliseconds
    text: str  # Original / current text
    translated_text: Optional[str] = None  # Translated & cleaned text

    @property
    def start_str(self) -> str:
        return pysubs2.time.ms_to_str(self.start_ms)

    @property
    def end_str(self) -> str:
        return pysubs2.time.ms_to_str(self.end_ms)


class SubtitleParser:
    """Handles loading, parsing, manipulating, and saving subtitle files."""

    @staticmethod
    def _detect_encoding(file_path: Path) -> str:
        """Automatically detects file encoding (UTF-8, UTF-8-SIG, CP949, EUC-KR, Windows-1252, etc.)."""
        with open(file_path, "rb") as f:
            raw = f.read(65536)

        # 1. BOM checks
        if raw.startswith(b"\xef\xbb\xbf"):
            return "utf-8-sig"
        if raw.startswith(b"\xff\xfe"):
            return "utf-16-le"
        if raw.startswith(b"\xfe\xff"):
            return "utf-16-be"

        # 2. Try UTF-8 full read
        try:
            with open(file_path, "rb") as f:
                f.read().decode("utf-8")
            return "utf-8"
        except UnicodeDecodeError:
            pass

        # 3. Try charset_normalizer
        try:
            import charset_normalizer
            match = charset_normalizer.from_path(file_path).best()
            if match and match.encoding:
                return match.encoding
        except Exception:
            pass

        # 4. Fallback common encodings
        for enc in ["cp949", "euc-kr", "windows-1252", "latin-1", "iso-8859-1"]:
            try:
                with open(file_path, "rb") as f:
                    f.read().decode(enc)
                return enc
            except Exception:
                continue

        return "utf-8"

    @staticmethod
    def load(file_path: str | Path) -> List[SubtitleItem]:
        """Loads a subtitle file (.srt, .vtt, .ass, etc.) with automatic encoding detection."""
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Subtitle file not found: {path}")

        encoding = SubtitleParser._detect_encoding(path)
        try:
            subs = pysubs2.load(str(path), encoding=encoding)
        except Exception:
            # Fallback with cp949 or latin-1 if primary detection fails
            try:
                subs = pysubs2.load(str(path), encoding="cp949")
            except Exception:
                subs = pysubs2.load(str(path), encoding="latin-1")

        items = []
        for idx, event in enumerate(subs, start=1):
            text = event.text.replace(r"\N", "\n").replace(r"\n", "\n").strip()
            # Skip completely empty events
            if not text:
                continue
            items.append(
                SubtitleItem(
                    index=idx,
                    start_ms=event.start,
                    end_ms=event.end,
                    text=text,
                )
            )
        return items

    @staticmethod
    def save_srt(
        items: List[SubtitleItem],
        output_path: str | Path,
        use_translated: bool = True,
    ) -> Path:
        """
        Saves SubtitleItem list to a clean .srt format with UTF-8 encoding.
        """
        out_path = Path(output_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        subs = pysubs2.SSAFile()
        for item in items:
            text_to_save = item.translated_text if (use_translated and item.translated_text) else item.text
            # Clean up line breaks
            text_to_save = text_to_save.replace("\r\n", "\n").replace("\n", r"\N")
            event = pysubs2.SSAEvent(
                start=item.start_ms,
                end=item.end_ms,
                text=text_to_save,
            )
            subs.append(event)

        subs.save(str(out_path), format_="srt", encoding="utf-8")
        return out_path

    @staticmethod
    def from_whisper_segments(segments) -> List[SubtitleItem]:
        """Converts faster-whisper segment generator/list into SubtitleItem list."""
        items = []
        for idx, segment in enumerate(segments, start=1):
            start_ms = int(segment.start * 1000)
            end_ms = int(segment.end * 1000)
            text = segment.text.strip()
            if not text:
                continue
            items.append(
                SubtitleItem(
                    index=idx,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    text=text,
                )
            )
        return items
