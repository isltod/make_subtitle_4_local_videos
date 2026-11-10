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
    """Uses Google Gemini API via high-speed direct REST with automatic fallback and retry."""

    AVAILABLE_MODELS = [
        "gemini-3.1-flash-lite",
        "gemini-flash-lite-latest",
        "gemini-3.5-flash-lite",
        "gemini-3.7-flash",
        "gemini-3.8-flash",
        "gemini-3.5-flash",
    ]

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
        self.chunk_size = chunk_size or SUBTITLE_CHUNK_SIZE
        self.model_name = model_name or "gemini-3.1-flash-lite"

        # Prioritize requested model, then fallback through available models
        models = list(self.AVAILABLE_MODELS)
        clean_model_name = self.model_name.split()[0].strip()
        if clean_model_name in models:
            models.remove(clean_model_name)
            models.insert(0, clean_model_name)
        elif clean_model_name:
            models.insert(0, clean_model_name)
        self.models_to_try = models

    def process_subtitles(
        self,
        items: List[SubtitleItem],
        custom_instructions: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[SubtitleItem]:
        """
        Translates foreign text to Korean and cleans/corrects spacing and grammar.
        Includes missing item recovery and truncation guards.
        """
        if not items:
            return items

        total_items = len(items)
        if progress_callback:
            progress_callback(0, total_items, f"총 {total_items}개 자막 항목 번역 및 교정 시작...")

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

            # 1st attempt: Call Gemini API with automatic fallback & retry
            translated_dict = self._call_gemini_rest(user_prompt)

            # Truncation Guard: Repeat recovery until ALL items in the chunk are translated (up to 3 recovery rounds)
            recovery_rounds = 0
            while recovery_rounds < 3:
                missing_items = [item for item in chunk if item.index not in translated_dict]
                if not missing_items:
                    break

                recovery_rounds += 1
                print(f"ℹ️ 청크 {chunk_idx}: {len(chunk)}개 중 {len(missing_items)}개 자막 누락 감지. 자동 복구 재요청 중... (시도 {recovery_rounds}/3)")

                missing_lines = []
                for item in missing_items:
                    clean_text = item.text.replace("\n", " ").strip()
                    missing_lines.append(f"{item.index} | {clean_text}")

                recovery_prompt = (
                    "앞서 누락된 자막 번호들입니다. 각 자막 번호(ID)를 변경하지 말고 주어진 번호 그대로 1:1 매칭하여 번역 및 교정하십시오:\n\n"
                    + "\n".join(missing_lines)
                )
                if custom_instructions:
                    recovery_prompt += f"\n\n[추가 요청사항]\n{custom_instructions}"

                time.sleep(1.0)
                recovered_dict = self._call_gemini_rest(recovery_prompt)
                if recovered_dict:
                    translated_dict.update(recovered_dict)
                else:
                    time.sleep(2.0)

            # Map responses back to items
            untranslated_in_chunk = 0
            for item in chunk:
                if item.index in translated_dict:
                    item.translated_text = translated_dict[item.index]
                else:
                    item.translated_text = item.text
                    untranslated_in_chunk += 1

            if untranslated_in_chunk > 0:
                print(f"⚠️ 청크 {chunk_idx}: {untranslated_in_chunk}개 항목 번역 실패로 원문 유지됨.")

            # Gentle pacing to avoid burst rate limits
            time.sleep(0.6)

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

    def _call_gemini_rest(self, user_content: str, timeout: int = 45) -> dict[int, str]:
        """Direct REST call to Google Gemini with exponential backoff and multi-model fallback."""
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
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
            ],
        }

        for model in self.models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"

            # Retry up to 2 times per model with backoff on 429 or timeout
            max_retries = 2
            for attempt in range(max_retries + 1):
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
                    elif resp.status_code == 429:
                        # Rate limit: Wait and retry if attempts remain
                        if attempt < max_retries:
                            wait_sec = 2.5 * (attempt + 1)
                            print(f"ℹ️ [{model}] 429 Rate Limit 감지. {wait_sec:.1f}초 후 재시도 중... (시도 {attempt+1}/{max_retries})")
                            time.sleep(wait_sec)
                            continue
                        else:
                            print(f"ℹ️ [{model}] 429 재시도 초과. 다음 모델로 전환합니다.")
                            break
                    elif resp.status_code in (500, 502, 503, 504):
                        if attempt < max_retries:
                            time.sleep(2.0)
                            continue
                        else:
                            print(f"ℹ️ [{model}] 서버 에러({resp.status_code}). 다음 모델로 전환합니다.")
                            break
                    else:
                        # 400, 404, etc: Unrecoverable on this model, switch immediately
                        status = resp.status_code
                        print(f"ℹ️ [{model}] 응답 코드 {status}. 다음 추천 모델로 신속히 전환합니다...")
                        break
                except requests.exceptions.Timeout:
                    if attempt < max_retries:
                        print(f"ℹ️ [{model}] 요청 시간 초과. 재시도 중... (시도 {attempt+1}/{max_retries})")
                        time.sleep(1.5)
                        continue
                    else:
                        print(f"ℹ️ [{model}] 타임아웃 지속. 다음 모델로 전환합니다.")
                        break
                except Exception as e:
                    print(f"ℹ️ [{model}] 요청 에러 ({e}). 다음 모델 시도 중...")
                    break

        print("⚠️ 모든 모델 호출 실패. 원본 텍스트를 보존합니다.")
        return {}

    @staticmethod
    def _parse_gemini_response(text: str) -> dict[int, str]:
        """Parses `ID | text` lines from Gemini response, preserving multi-line subtitle dialogue cleanly."""
        result = {}
        # Matches '1 | text' or '1: text' or '1. text' or '[1] text'
        line_pattern = re.compile(r"^\s*\[?(\d+)\]?\s*(?:\||:|-|\.)\s*(.+)$")
        # Matches bare numbers like '6' or '6 |' or '6.' with no substantive text
        bare_number_pattern = re.compile(r"^\s*\[?(\d+)\]?\s*(?:\||:|-|\.)?\s*$")

        current_idx = None

        for raw_line in text.splitlines():
            line = raw_line.strip()
            # Ignore completely empty lines and markdown fences
            if not line or line.startswith("```"):
                continue

            match = line_pattern.match(line)
            if match:
                current_idx = int(match.group(1))
                content = match.group(2).strip()
                result[current_idx] = content
            elif bare_number_pattern.match(line):
                # Bare number on its own line: treat as index indicator, not continuation text
                current_idx = int(bare_number_pattern.match(line).group(1))
            elif current_idx is not None:
                # Append continuation lines (e.g. second speaker dialogue starting with -)
                if current_idx in result:
                    result[current_idx] += f"\n{line}"
                else:
                    result[current_idx] = line

        return result


# Quick singleton instance
cleaner_translator = GeminiSubtitleCleaner()
