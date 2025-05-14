# baseballCompanion

This app runs locally and allows you to ask natural language questions about the current state of baseball based on recent YouTube analysis videos. It uses Whisper for transcription, SentenceTransformers for embedding, FAISS as a vector database, and llama.cpp for local LLM inference with RAG (Retrieval-Augmented Generation).



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

