import os
import sys

# Configure UTF-8 for console output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()

print("=" * 60)
print("🔍 Environment & Hardware Diagnostic Test")
print("=" * 60)

# Check Python
print(f"🐍 Python Executable: {sys.executable}")
print(f"🐍 Python Version: {sys.version.split()[0]}")

# 2. Check PyTorch & GPU / CUDA
print("-" * 60)
print("🎮 Checking GPU & PyTorch CUDA acceleration...")
try:
    import torch
    print(f"✅ PyTorch Version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        print(f"GPU Count: {gpu_count}")
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            vram = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            print(f"  - GPU [{i}]: {name} ({vram:.2f} GB VRAM)")
    else:
        print("⚠️ CUDA is not available in PyTorch.")
except Exception as e:
    print(f"❌ PyTorch Check Error: {e}")

# 3. Check faster-whisper & ctranslate2
print("-" * 60)
print("🎙️ Checking faster-whisper (STT Engine)...")
try:
    import faster_whisper
    import ctranslate2
    print(f"✅ faster-whisper Version: {faster_whisper.__version__}")
    dev_count = ctranslate2.get_cuda_device_count()
    cuda_types = ctranslate2.get_supported_compute_types("cuda") if dev_count > 0 else []
    print(f"CTranslate2 CUDA devices: {dev_count}, Supported compute types: {cuda_types}")
except Exception as e:
    print(f"ℹ️ faster-whisper test status: {e}")

# 4. Check Gemini API Connection
print("-" * 60)
print("✨ Checking Google Gemini API...")
gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key or gemini_key.strip() == "" or "your_gemini_api_key_here" in gemini_key:
    print("⚠️ GEMINI_API_KEY is not set or placeholder in .env")
else:
    masked_key = gemini_key[:6] + "..." + gemini_key[-4:] if len(gemini_key) > 10 else "***"
    print(f"🔑 API Key Found: {masked_key}")
    try:
        from google import genai
        client = genai.Client(api_key=gemini_key)
        
        # Test generation with the latest Gemini 3.7 Flash
        target_model = 'gemini-3.7-flash'
        print(f"🚀 Testing Target Model: [{target_model}]...")
        response = client.models.generate_content(
            model=target_model,
            contents='안녕! Gemini 3.7 Flash API가 정상 연결되었는지 한 문장으로 확인해줘.',
        )
        print(f"✅ Gemini API Test Succeeded with model: [{target_model}]")
        print(f"🤖 Response: {response.text.strip()}")
    except Exception as e:
        print(f"❌ Gemini API Test Failed: {e}")

print("=" * 60)
