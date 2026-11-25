import json
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
    """Uses Google Gemini API via structured JSON REST output with ID-shift protection, automatic fallback, and retry."""

    AVAILABLE_MODELS = [
        "gemini-3.1-flash-lite",
        "gemini-flash-lite-latest",
        "gemini-3.5-flash-lite",
        "gemini-3.7-flash",
        "gemini-3.8-flash",
        "gemini-3.5-flash",
    ]

    RESPONSE_SCHEMA = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "id": {"type": "INTEGER"},
                "text": {"type": "STRING"},
            },
            "required": ["id", "text"],
        },
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = GEMINI_MODEL,
        chunk_size: int = SUBTITLE_CHUNK_SIZE,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "") or GEMINI_API_KEY
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
        Uses strict JSON schema output + recursive split-on-mismatch to guarantee zero ID shifts.
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
                    f"자막 번역/교정 중... ({chunk_idx}/{total_chunks} 단계, {len(chunk)}개 라인 전송)",
                )

            translated_dict = self._translate_chunk_safe(
                chunk=chunk,
                custom_instructions=custom_instructions,
            )

            # Map responses back to items
            untranslated_in_chunk = 0
            for item in chunk:
                if item.index in translated_dict and translated_dict[item.index].strip():
                    item.translated_text = translated_dict[item.index].strip()
                else:
                    item.translated_text = item.text
                    untranslated_in_chunk += 1

            if untranslated_in_chunk > 0:
                print(f"⚠️ 청크 {chunk_idx}: {untranslated_in_chunk}개 항목 번역 실패로 원문 유지됨.")

            # Gentle pacing to avoid burst rate limits
            time.sleep(0.5)

            processed_count = min(chunk_idx * chunk_size, total_items)
            if progress_callback:
                progress_callback(
                    processed_count,
                    total_items,
                    f"자막 번역 완료 ({processed_count}/{total_items})",
                )

        if progress_callback:
            progress_callback(total_items, total_items, "✅ 모든 자막 번역 및 교정 완료!")

        return items

    def _translate_chunk_safe(
        self,
        chunk: List[SubtitleItem],
        custom_instructions: Optional[str] = None,
        depth: int = 0,
    ) -> dict[int, str]:
        """
        Translates a chunk with strict 1:1 ID verification.
        If ANY ID count/set mismatch occurs (indicating a potential ID shift from merged lines or truncation),
        discards the tainted chunk output and recursively splits the chunk in half instead of patching tail IDs.
        """
        expected_ids = [item.index for item in chunk]
        expected_set = set(expected_ids)

        payload_items = [
            {"id": item.index, "text": item.text.replace("\n", " ").strip()}
            for item in chunk
        ]
        user_prompt = (
            f"다음 {len(chunk)}개의 자막 항목을 입력과 동일한 id 순서로 정확히 1:1 매칭하여 "
            f"한국어로 번역 및 교정해 주십시오 (인접 항목 병합, 누락, 번호 이동 절대 금지):\n\n"
            + json.dumps(payload_items, ensure_ascii=False, indent=2)
        )
        if custom_instructions:
            user_prompt += f"\n\n[추가 요청사항]\n{custom_instructions}"

        translated_dict = self._call_gemini_rest(user_prompt)

        returned_set = set(translated_dict.keys())
        all_non_empty = all(
            bool(translated_dict.get(idx, "").strip()) for idx in expected_ids
        )

        # Perfect 1:1 match!
        if returned_set == expected_set and all_non_empty:
            return translated_dict

        # Mismatch detected: could be an ID shift (merged lines) or early truncation.
        # Split in half recursively so shifted lines never survive into the final SRT.
        if len(chunk) > 6 and depth < 4:
            print(
                f"ℹ️ ID 개수/매칭 불일치 감지 (요청 {len(chunk)}개 / 일치 {len(returned_set & expected_set)}개). "
                f"번호 밀림(ID Shift) 방지를 위해 반으로 분할하여 재번역합니다..."
            )
            mid = len(chunk) // 2
            time.sleep(0.5)
            left = self._translate_chunk_safe(chunk[:mid], custom_instructions, depth + 1)
            time.sleep(0.4)
            right = self._translate_chunk_safe(chunk[mid:], custom_instructions, depth + 1)
            merged = {}
            merged.update(left)
            merged.update(right)
            return merged

        # Small chunk (<= 6 items): recover any missing IDs directly
        result = {
            idx: translated_dict[idx]
            for idx in expected_ids
            if idx in translated_dict and translated_dict[idx].strip()
        }
        for _ in range(2):
            missing = [item for item in chunk if item.index not in result]
            if not missing:
                break
            time.sleep(0.8)
            missing_payload = [
                {"id": item.index, "text": item.text.replace("\n", " ").strip()}
                for item in missing
            ]
            rec_prompt = (
                "누락된 자막 항목입니다. 주어진 id 그대로 정확히 1:1 매칭하여 한국어로 번역하십시오:\n\n"
                + json.dumps(missing_payload, ensure_ascii=False, indent=2)
            )
            if custom_instructions:
                rec_prompt += f"\n\n[추가 요청사항]\n{custom_instructions}"
            rec_dict = self._call_gemini_rest(rec_prompt)
            for item in missing:
                if item.index in rec_dict and rec_dict[item.index].strip():
                    result[item.index] = rec_dict[item.index]

        return result

    def _call_gemini_rest(self, user_content: str, timeout: int = 45) -> dict[int, str]:
        """Direct REST call to Google Gemini with JSON schema enforcement, exponential backoff, and fallback."""
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
                "temperature": 0.2,
                "responseMimeType": "application/json",
                "responseSchema": self.RESPONSE_SCHEMA,
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
        """Parses structured JSON array `[{"id": int, "text": str}]`, with text fallback."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()

        # 1. Primary: Strict JSON parsing
        try:
            data = json.loads(cleaned)
            if isinstance(data, list):
                result: dict[int, str] = {}
                for entry in data:
                    if isinstance(entry, dict) and "id" in entry and "text" in entry:
                        idx = int(entry["id"])
                        val = str(entry["text"]).strip()
                        if idx in result:
                            # Duplicate ID indicates corruption/shift -> discard so caller splits safely
                            return {}
                        result[idx] = val
                if result:
                    return result
        except (ValueError, TypeError):
            pass

        # 2. Fallback: Legacy `ID | text` parsing
        result = {}
        line_pattern = re.compile(r"^\s*\[?(\d+)\]?\s*(?:\||:|-|\.)\s*(.+)$")
        bare_number_pattern = re.compile(r"^\s*\[?(\d+)\]?\s*(?:\||:|-|\.)?\s*$")

        current_idx = None
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("```"):
                continue

            match = line_pattern.match(line)
            if match:
                current_idx = int(match.group(1))
                content = match.group(2).strip()
                result[current_idx] = content
            elif bare_number_pattern.match(line):
                current_idx = int(bare_number_pattern.match(line).group(1))
            elif current_idx is not None:
                if current_idx in result:
                    result[current_idx] += f"\n{line}"
                else:
                    result[current_idx] = line

        return result


# Quick singleton instance
cleaner_translator = GeminiSubtitleCleaner()
