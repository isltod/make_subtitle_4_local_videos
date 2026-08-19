import os
import re
import time
import requests
from typing import List, Optional, Callable

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    SUBTITLE_CHUNK_SIZE,
    TRANSLATION_SYSTEM_PROMPT,
)
from core.subtitle_parser import SubtitleItem


class GeminiSubtitleCleaner:
    """Uses Google Gemini API via high-speed direct REST with automatic fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = GEMINI_MODEL,
        chunk_size: int = SUBTITLE_CHUNK_SIZE,
    ):
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            raise ValueError(
                "Gemini API Key가 설정되지 않았습니다. .env 파일에 GEMINI_API_KEY를 입력해 주세요."
            )
        self.model_name = "gemini-3.5-flash"
        self.chunk_size = 100
        # gemini-3.5-flash and gemini-3.5-flash-lite respond in 2-3s without server holds
        self.models_to_try = [
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
        ]

    def process_subtitles(
        self,
        items: List[SubtitleItem],
        custom_instructions: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[SubtitleItem]:
        """
        Translates foreign text to Korean and cleans/corrects spacing and grammar.
        """
        if not items:
            return items

        total_items = len(items)
        if progress_callback:
            progress_callback(0, total_items, f"총 {total_items}개 자막 항목 번역 및 교정 시작...")

        # Process in chunks (150-250 lines per chunk for high context & speed)
        chunk_size = self.chunk_size
        chunks = [items[i : i + chunk_size] for i in range(0, total_items, chunk_size)]
        total_chunks = len(chunks)

        for chunk_idx, chunk in enumerate(chunks, start=1):
            if progress_callback:
                progress_callback(
                    (chunk_idx - 1) * chunk_size,
                    total_items,
                    f"자막 번역/교정 중... ({chunk_idx}/{total_chunks} 단계, {len(chunk)}개 라인 전송)"
                )

            # Prepare formatted prompt with IDs
            lines = []
            for item in chunk:
                clean_text = item.text.replace("\n", " ").strip()
                lines.append(f"{item.index} | {clean_text}")

            user_prompt = "다음 자막 목록을 규칙에 맞추어 한국어로 번역 및 맞춤법/띄어쓰기를 교정해 주십시오:\n\n" + "\n".join(lines)
            if custom_instructions:
                user_prompt += f"\n\n[추가 요청사항]\n{custom_instructions}"

            # Call Gemini API with automatic fallback
            translated_dict = self._call_gemini_rest(user_prompt)

            # Map responses back to items
            for item in chunk:
                if item.index in translated_dict:
                    item.translated_text = translated_dict[item.index]
                else:
                    item.translated_text = item.text

            processed_count = min(chunk_idx * chunk_size, total_items)
            if progress_callback:
                progress_callback(
                    processed_count,
                    total_items,
                    f"자막 번역 완료 ({processed_count}/{total_items})"
                )

        if progress_callback:
            progress_callback(total_items, total_items, "✅ 모든 자막 번역 및 교정 완료!")

        return items

    def _call_gemini_rest(self, user_content: str, timeout: int = 35) -> dict[int, str]:
        """Direct REST call to Google Gemini with fast fallback on 429/503."""
        payload = {
            "systemInstruction": {
                "parts": [{"text": TRANSLATION_SYSTEM_PROMPT}]
            },
            "contents": [
                {
                    "parts": [{"text": user_content}]
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
            }
        }

        for model in self.models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
            try:
                resp = requests.post(url, json=payload, timeout=timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            text = parts[0].get("text", "")
                            parsed = self._parse_gemini_response(text)
                            if parsed:
                                return parsed
                else:
                    # If 429 or 503 or 404, quickly try next model
                    status = resp.status_code
                    print(f"ℹ️ [{model}] 응답 코드 {status}. 다음 추천 모델로 신속히 전환합니다...")
            except Exception as e:
                print(f"ℹ️ [{model}] 요청 에러 ({e}). 다음 모델 시도 중...")

        print("⚠️ 모든 모델 호출 실패. 원본 텍스트를 보존합니다.")
        return {}

    @staticmethod
    def _parse_gemini_response(text: str) -> dict[int, str]:
        """Parses `ID | text` lines from Gemini response."""
        result = {}
        pattern = re.compile(r"^\s*\[?(\d+)\]?\s*(?:\||:|-|\.)\s*(.+)$")

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            match = pattern.match(line)
            if match:
                idx = int(match.group(1))
                content = match.group(2).strip()
                result[idx] = content

        return result


# Quick singleton instance
cleaner_translator = GeminiSubtitleCleaner()
