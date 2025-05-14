import os
import argparse
from faster_whisper import WhisperModel
import yt_dlp


def download_audio(youtube_url, out_dir="downloads"):
    os.makedirs(out_dir, exist_ok=True)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{out_dir}/%(id)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
        }]
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        return f"{out_dir}/{info['id']}.mp3", info['id']


def transcribe(audio_path, model_size="medium", output_dir="transcripts"):
    model = WhisperModel(model_size, compute_type="int8")
    segments, _ = model.transcribe(audio_path)
    os.makedirs(output_dir, exist_ok=True)
    text_chunks = []
    for seg in segments:
        text_chunks.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text
        })

    output_path = os.path.join(output_dir, os.path.basename(audio_path).replace(".mp3", ".txt"))
    with open(output_path, "w") as f:
        for chunk in text_chunks:
            f.write(f"[{chunk['start']:.2f} - {chunk['end']:.2f}] {chunk['text']}\n")

    print(f"Transcription saved to: {output_path}")
    return None


def main():
    parser = argparse.ArgumentParser(description="Download and transcribe audio from a YouTube video.")
    parser.add_argument('--url', type=str, required=True, help='YouTube video URL')
    parser.add_argument('--model-size', type=str, default='medium', help='Whisper model size (e.g., tiny, base, small, medium, large)')
    args = parser.parse_args()

    print(f"Downloading audio from: {args.url}")
    audio_path, video_id = download_audio(args.url)

    print(f"Transcribing using model size: {args.model_size}")
    transcribe(audio_path, model_size=args.model_size)


if __name__ == "__main__":
    main()