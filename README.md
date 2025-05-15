# baseballCompanion

This app runs locally and allows you to ask natural language questions about the current state of baseball based on recent YouTube analysis videos. It uses Whisper for transcription, SentenceTransformers for embedding, FAISS as a vector database, and llama.cpp for local LLM inference with RAG (Retrieval-Augmented Generation).


## Tools:

- **faster-whisper** for super fast transcription of videos to audio
- yt-dlp as a download engine for videos
- **llama.cpp** as an inference engine with flexible model backends and can run on apple silicon
- **mistral-7b-instruct** as the base model
- 



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
brew install llama.cpp


# cd llama.cpp
# make LLAMA_METAL=1
mkdir -p models
# Download a quantized GGUF model (example: Mistral 7B)
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf -P models/

# test in conversation mode:
llama-cli -m models/mistral-7b-instruct-v0.1.Q4_K_M.gguf -cnv --chat-template chatml
```


## Tools


### Faster Whisper

Whisper for audio transcription into text:

[Faster Whisper Library ](https://github.com/SYSTRAN/faster-whisper)

faster-whisper is a reimplementation of OpenAI's Whisper model using CTranslate2, which is a fast inference engine for Transformer models.

This implementation is up to 4 times faster than openai/whisper for the same accuracy while using less memory. The efficiency can be further improved with 8-bit quantization on both CPU and GPU.

`pip install faster-whisper`

```python
from faster_whisper import WhisperModel

model_size = "large-v3"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe("audio.mp3", beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
```
### YT-DLP


[YT-DLP ](https://github.com/yt-dlp/yt-dlp)

yt-dlp is a feature-rich command-line audio/video downloader with support for thousands of sites. The project is a fork of youtube-dl based on the now inactive youtube-dlc.

### Usage

* To get the transcription engine running:

ToDo: this needs to be automated to keep the DB fresh. 

```bash
python engine.py --url https://youtube.com/watch?v=abc123 --model-size small
```

### LLama.cpp

Let's use pure c++ inference from `https://github.com/ggml-org/llama.cpp`. This project also supports multimodal inputs now, so we can eventually extend this to pure video for analysis. It supports LLaVA 1.5 models, Qwen2-VL, Moondream (a personal favorite), and some ~15 other multi-modal models. For text-only models, it supports Gemma, Mamba Grok-1, Gpts, Deepseek models, Mistral models, some Mixtral Mixture of Experts, along some ~50 others. 



"The main goal of llama.cpp is to enable LLM inference with minimal setup and state-of-the-art performance on a wide range of hardware - locally and in the cloud."

## Execution and Plan

### Setup & Proof of Concept 

	•	Set up llama.cpp locally.

	•	Run Whisper on a test video and store transcript.

	•	Build embedding + vector store prototype.

	•	Manual prompt + RAG integration with llama.cpp.

### Backend Pipeline 

	•	Automate YouTube → transcript → embedding → store.

	•	Script for refreshing content weekly/daily.

	•	Test with multiple baseball YouTube episodes.

### Desktop App 

	•	Develop UI for search/QA and sentiment display.

	•	Connect LLM and vector DB backend to UI.

	•	Local persistent state and vector DB management.




### ToDo:

	•	Improve prompt templates for answer relevance.
	•	Add UI features like transcript preview, sentiment summaries, model stats.
	•	Test usability, performance, edge cases.
	•	Stream video transcription and real-time updates.
	•	Speaker diarization and source tracking.
	•	Natural language summarization of multiple videos.
	•	Offline video analysis and content tagging.


**Disclaimer** --> This is an educational tool to showcase how to work with llms and RAG applications, not for commercial use. 

