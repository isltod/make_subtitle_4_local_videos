import argparse
import sys
from pathlib import Path

# Configure UTF-8 for console output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config import GEMINI_MODEL, WHISPER_MODEL_SIZE
from core.pipeline import SubtitlePipeline
from core.transcriber import WhisperTranscriber
from core.cleaner_translator import GeminiSubtitleCleaner


def main():
    parser = argparse.ArgumentParser(
        description="🎬 로컬 동영상 고품질 한국어 자막 자동 생성 및 교정 도구 (RTX 3090 Whisper + Gemini 3.7 Flash)"
    )

    parser.add_argument(
        "input",
        nargs="?",
        help="처리할 동영상 또는 자막 파일 경로 (예: D:\\Videos\\movie.mp4)",
    )
    parser.add_argument(
        "--folder",
        "-f",
        help="일괄 처리할 동영상 폴더 경로 (예: D:\\Videos)",
    )
    parser.add_argument(
        "--sub",
        "-s",
        help="기존 외국어/오탈자 자막 파일 경로 (동영상과 별도로 지정 시)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="생성할 한국어 자막 저장 경로 (지정하지 않으면 원본 영상 옆에 .ko.srt 로 저장)",
    )
    parser.add_argument(
        "--lang",
        "-l",
        help="원본 음성 언어 힌트 (예: en, ja, zh, ko - 미지정 시 자동 감지)",
    )
    parser.add_argument(
        "--instructions",
        "-i",
        help="Gemini 번역 시 추가 전달할 프롬프트 지시사항 (예: '등장인물 간 존댓말 유지, 의학 전문용어 자연스럽게 번역')",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="폴더 처리 시 하위 디렉토리까지 재귀적으로 탐색",
    )
    parser.add_argument(
        "--model",
        default=GEMINI_MODEL,
        help=f"사용할 Gemini 모델명 (기본값: {GEMINI_MODEL})",
    )
    parser.add_argument(
        "--whisper-model",
        default=WHISPER_MODEL_SIZE,
        help=f"사용할 Whisper 모델 크기 (기본값: {WHISPER_MODEL_SIZE})",
    )

    args = parser.parse_args()

    # Create custom pipeline if models were overridden
    stt = WhisperTranscriber(model_size=args.whisper_model)
    cleaner = GeminiSubtitleCleaner(model_name=args.model)
    pipeline = SubtitlePipeline(stt=stt, cleaner=cleaner)

    print("=" * 65)
    print("🎬 Make Subtitle for Local Videos (AI 자막 파이프라인)")
    print(f"⚡ Whisper STT 모델: [{args.whisper_model}] (GPU: CUDA)")
    print(f"✨ Gemini 번역/교정 모델: [{args.model}]")
    print("=" * 65)

    if args.folder:
        folder_path = Path(args.folder).resolve()
        if not folder_path.exists() or not folder_path.is_dir():
            print(f"❌ 폴더를 찾을 수 없습니다: {folder_path}")
            sys.exit(1)

        print(f"📂 폴더 일괄 처리 모드: {folder_path}")
        results = pipeline.process_folder(
            folder_path=folder_path,
            recursive=args.recursive,
            source_language=args.lang,
            custom_instructions=args.instructions,
        )
        print(f"\n🎉 일괄 처리 완료! 총 {len(results)}개 자막 파일 생성됨.")

    elif args.input:
        input_path = Path(args.input).resolve()
        if not input_path.exists():
            print(f"❌ 파일을 찾을 수 없습니다: {input_path}")
            sys.exit(1)

        print(f"🎯 단일 파일 처리 모드: {input_path.name}")
        out_file = pipeline.process_file(
            input_path=input_path,
            existing_subtitle_path=args.sub,
            output_srt_path=args.output,
            source_language=args.lang,
            custom_instructions=args.instructions,
        )
        print(f"\n🎉 완료! 생성된 자막 파일: {out_file}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
