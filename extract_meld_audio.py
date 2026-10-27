import os
import subprocess
import glob

print("--- MELD Dataset Audio Extractor ---", flush=True)

# Define your input MELD directory (update this to point to where your MELD .mp4 files are located)
meld_source_dir = 'MELD-master'  # e.g., folder containing train/dev/test or video clips
output_audio_dir = 'audio'

if not os.path.exists(meld_source_dir):
    print(
        f"[ERROR] Source directory '{meld_source_dir}' not found. Please update 'meld_source_dir' to point to your MELD files.")
else:
    os.makedirs(output_audio_dir, exist_ok=True)

    # Find all .mp4 files recursively
    mp4_files = glob.glob(os.path.join(meld_source_dir, '**', '*.mp4'), recursive=True)
    print(f"[PROCESSING] Found {len(mp4_files)} video files. Converting to .wav...", flush=True)

    success_count = 0
    for idx, mp4_path in enumerate(mp4_files):
        filename = os.path.basename(mp4_path)
        base_name = os.path.splitext(filename)[0]
        output_wav_path = os.path.join(output_audio_dir, f"{base_name}.wav")

        # Skip if already converted
        if os.path.exists(output_wav_path):
            success_count += 1
            continue

        # Run ffmpeg command to extract audio: 22050Hz, Mono, PCM 16-bit
        cmd = [
            'ffmpeg', '-y', '-i', mp4_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '22050', '-ac', '1', output_wav_path
        ]

        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"[WARNING] Failed to convert {filename}: {e}")

        if (idx + 1) % 500 == 0:
            print(f"  Converted {idx + 1}/{len(mp4_files)} files...", flush=True)

    print(
        f"\n[SUCCESS] Conversion complete! Successfully extracted {success_count} .wav audio files into '{output_audio_dir}/'.",
        flush=True)
