# Local Baseball Companion App (RAG + Whisper + llama.cpp)




---

## 0. Prerequisites

Install the required tools and packages:

```bash
# Install FFmpeg
brew install ffmpeg  # or sudo apt install ffmpeg

# Install yt-dlp
pip install yt-dlp

# Install Whisper
pip install faster-whisper

# Install sentence-transformers and FAISS
pip install sentence-transformers faiss-cpu

# Clone llama.cpp and compile
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make LLAMA_CUBLAS=1  # or just 'make' for CPU

# Download a quantized GGUF model (example: Mistral 7B)
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf -P models/
```

Test llama.cpp:

```bash
./main -m models/mistral-7b-instruct-v0.1.Q4_K_M.gguf -p "Who won the 2023 World Series?"
```

---

## 1. Download and Transcribe YouTube Videos

```python
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
```

---

## 2. Generate and Store Embeddings with FAISS

```python
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

model = SentenceTransformer("all-MiniLM-L6-v2")

def build_vector_index(chunks, video_id, save_dir="faiss_index"):
    texts = [c['text'] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(save_dir, exist_ok=True)
    faiss.write_index(index, f"{save_dir}/{video_id}.index")
    with open(f"{save_dir}/{video_id}_meta.pkl", "wb") as f:
        pickle.dump(chunks, f)
```

---

## 3. Search and Perform RAG with llama.cpp

### 3.1 Search Top-K Transcript Chunks

```python
def search(query, video_id, save_dir="faiss_index"):
    index = faiss.read_index(f"{save_dir}/{video_id}.index")
    with open(f"{save_dir}/{video_id}_meta.pkl", "rb") as f:
        chunks = pickle.load(f)
    
    query_vec = model.encode([query])
    D, I = index.search(query_vec, k=5)
    top_k_chunks = [chunks[i]['text'] for i in I[0]]
    return top_k_chunks
```

### 3.2 Format Prompt and Call llama.cpp

```python
import subprocess

def call_llama(context, query, model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"):
    prompt = f"""### Question:\n{query}\n\n### Context:\n{context}\n\n### Answer:\n"""
    cmd = [
        "./llama.cpp/main",
        "-m", model_path,
        "-p", prompt
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout
```

---

## 4. Optional: GUI with Python (Flet)

Install Flet:

```bash
pip install flet
```

GUI app:

```python
import flet as ft

def main(page):
    def on_submit(e):
        results = search(query_box.value, "video_id_here")
        response = call_llama("\n".join(results), query_box.value)
        output.controls.append(ft.Text(response))
        page.update()

    query_box = ft.TextField(label="Ask baseball something...")
    output = ft.Column()
    page.add(query_box, ft.ElevatedButton("Submit", on_click=on_submit), output)

ft.app(target=main)
```

---

## 5. Example End-to-End Script

```python
# Step 1: Download and transcribe
audio_path, vid_id = download_audio("https://youtube.com/watch?v=abc123")
chunks = transcribe(audio_path)

# Step 2: Build FAISS index
build_vector_index(chunks, vid_id)

# Step 3: Search and QA
top_k = search("Who are the top pitchers this week?", vid_id)
answer = call_llama("\n".join(top_k), "Who are the top pitchers this week?")
print(answer)
```

---

## 6. Optional Enhancements

- Sentiment analysis using HuggingFace models (e.g., cardiffnlp/twitter-roberta-base-sentiment).
- Summarization with T5-small or another LLM.
- Tauri or Electron-based GUI for better UX.
- Scheduler to download and process new videos on a regular basis.