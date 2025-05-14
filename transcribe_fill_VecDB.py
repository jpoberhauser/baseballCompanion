from faster_whisper import WhisperModel
import yt_dlp
import os



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
    return text_chunks