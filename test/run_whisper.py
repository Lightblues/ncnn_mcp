import pathlib
import subprocess
import hashlib
from datetime import datetime
import random
import string

def transcribe_audio(file_path: str) -> str:
    """ Transcribe audio file to text using NCNN Whisper model.
    """
    # Get absolute paths
    DIR = pathlib.Path(__file__).parent.parent
    build_examples_dir = DIR / "model"
    data_dir = build_examples_dir / "data"
    
    # Create data directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename: date+hash+random.wav
    file_hash = hashlib.md5(pathlib.Path(file_path).read_bytes()).hexdigest()[:8]
    date_str = datetime.now().strftime("%Y%m%d")
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    output_wav = f"{date_str}_{file_hash}_{random_suffix}.wav"
    output_wav_path = data_dir / output_wav
    
    # step1: ffmpeg - convert to PCM format and save to data directory
    ffmpeg_cmd = [
        "ffmpeg", "-i", file_path, "-vn", "-c:a", "pcm_s16le", "-ac", "1", "-ar", "16000", "-fflags", "bitexact", str(output_wav_path)
    ]
    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Converted audio saved to: {output_wav_path}")

    # step2: run whisper in ncnn/build/examples directory
    whisper_cmd = ["./whisper", str(output_wav_path)]
    result = subprocess.run(whisper_cmd, check=True, capture_output=True, text=True, cwd=str(build_examples_dir / "whisper"))

    # The whisper example prints status and the final transcript. Some binaries print to stderr.
    combined_output = "\n".join(filter(None, [result.stdout, result.stderr]))
    print(f"combinationed_output:\n{combined_output}")

    # Parse lines like: "text =  the the" and return the part after '=' trimmed
    for line in combined_output.splitlines():
        line = line.strip()
        if line.startswith("text ="):
            # split once on '=' and strip surrounding whitespace
            parts = line.split("=", 1)
            if len(parts) > 1:
                return parts[1].strip()

    # Fallback: return full combined output (trimmed)
    return combined_output.strip()

if __name__ == '__main__':
    # data from https://platform.openai.com/docs/guides/audio?example=audio-in
    audio_file = "data/alloy.wav"
    transcription = transcribe_audio(audio_file)
    print(f"Transcription Result: {transcription}")