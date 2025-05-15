# baseballCompanion

![baseballCompanion](assets/logo.png)

- [baseballCompanion](#baseballcompanion)
  - [Tools:](#tools)
  - [Example - before and after RAG](#example---before-and-after-rag)
  - [Example of Retrieval to Documents](#example-of-retrieval-to-documents)
  - [Prerequisites](#prerequisites)
    - [Faster Whisper](#faster-whisper)
    - [YT-DLP](#yt-dlp)
    - [LLama.cpp](#llamacpp)
    - [Embeddings](#embeddings)
      - [Question 1: Chunking](#question-1-chunking)
    - [LangChain Text Splitter](#langchain-text-splitter)
    - [Disclaimer: --\> This is an educational tool to showcase how to work with llms and RAG applications, not for commercial use.](#disclaimer----this-is-an-educational-tool-to-showcase-how-to-work-with-llms-and-rag-applications-not-for-commercial-use)


This app runs locally and allows you to ask natural language questions about the current state of baseball based on recent YouTube analysis videos. It uses Whisper for transcription, SentenceTransformers for embedding, FAISS as a vector database, and llama.cpp for local LLM inference with RAG (Retrieval-Augmented Generation).


## Tools:

- **faster-whisper** for super fast transcription of videos to audio
- yt-dlp as a download engine for videos
- **llama.cpp** as an inference engine with flexible model backends and can run on apple silicon
- **mistral-7b-instruct** as the base model
- **sentence_transformers** as the sentence embedding model
- **LangChain** for RAG, query construction, query refinement, etc..





## Example - before and after RAG

* The date is May 15, 2025. There are several trade rumors about Luis Robert to the Mets. Let's ask our assistant what people are saying about this:

* question: 'what are people saying today about trade rumors of Luis Robert to the Mets?'

Pre-context:

```
I don't have real-time information about specific trade rumors, but I can tell you that Luis Robert is a promising young outfielder who has been playing for the Chicago White Sox. He is known for his impressive performance at the plate and his speed on the bases. Some people may be interested in knowing what the Mets are thinking about the possibility of acquiring him, but it is important to note that nothing is certain until a trade is officially announced. Is there anything else you would like to know?<|im_end|>
```

Post-context:

```
Some people may be excited about the possibility of having him on the team, as he is known for his impressive performance at the plate and his speed on the bases. However, it is important to note that nothing is certain until a trade is officially announced. 

The video and podcast episode you mentioned also provide some context about Luis Robert's performance in recent seasons, including his batting average, 
on-base percentage, slugging, and OPS. It also mentions that he has been playing bad for two years, but that it is hard not to see what he did in previous seasons. 

In terms of value, a trade for Luis Robert would likely involve giving up a significant amount of assets, such as top prospects 
or established players. It is also worth considering the potential risk of injury, as Luis Robert has only played more than 101 games in one season in his entire career. 

Is there anything else you would like to know about this topic?
```

## Example of Retrieval to Documents

* query = "The New York Mets are very good"

```
query_embedding = model.encode(query)

# 4. Compute cosine similarities
cos_scores = util.cos_sim(query_embedding, embeddings)[0]  # shape: (3,)

# 5. Find most similar
most_similar_idx = cos_scores.argmax()
print(f"Query: {query}")
print(f"Most similar: {all_chunks[most_similar_idx]}")
```

Gives us: 

```
Most similar: York Mets, they went and got Juan Soto. They went and signed Francis Golden Door back. They've been bringing players in. 
Having the richest owner in the sport obviously helps that. If you go down the line, the Cubs made a big play for Tucker and the rest of the Astros team.
 They bring him over. Now, they're performing very well. The Dodgers have went out and signed everybody. 
 The Padres and all the trades they've made for Michael King, Dylan Cease, you know, they've done a lot. 
 Champions League of Giants have landed a bunch of free agents. That's been the difference. T
 hey've just got the good players when the good players were available. If you look at the American League team, 
 name me a team that's done that. Yeah. There isn't a team that's went out and got the big player. T
 he Yankees, they did that and they made it to the World Series. They went out and got Juan Soto. 
 Then this year, you know, they went out and made some trades for Belly for Williams. Those haven't been great, but
```



## Prerequisites

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


* To get the transcription engine running:

ToDo: this needs to be automated to keep the DB fresh. 

```bash
python engine.py --url https://youtube.com/watch?v=abc123 --model-size small
```

### LLama.cpp

Let's use pure c++ inference from [LLAMMA.cpp](https://github.com/ggml-org/llama.cpp). This project also supports multimodal inputs now, so we can eventually extend this to pure video for analysis. It supports LLaVA 1.5 models, Qwen2-VL, Moondream (a personal favorite), and some ~15 other multi-modal models. For text-only models, it supports Gemma, Mamba Grok-1, Gpts, Deepseek models, Mistral models, some Mixtral Mixture of Experts, along some ~50 others. 



"The main goal of llama.cpp is to enable LLM inference with minimal setup and state-of-the-art performance on a wide range of hardware - locally and in the cloud."

### Embeddings

* There's many ways to get embeddings. We can call models with an API key like Mistral, OpenAI, Anthropic embedding models. We can also go another route and try to run embeddings locally, with a local model, for example  [sentence_transformers](https://huggingface.co/sentence-transformers)


"Sentence Transformers (a.k.a. SBERT) is the go-to Python module for accessing, using, and training state-of-the-art embedding and reranker models. It can be used to compute embeddings using Sentence Transformer models or to calculate similarity scores using Cross-Encoder (a.k.a. reranker) models. This unlocks a wide range of applications, including semantic search, semantic textual similarity, and paraphrase mining."


```
model = SentenceTransformer("all-MiniLM-L6-v2")
```

#### Question 1: Chunking

* given a large document of pure text, how do we get chunks? 

* do we split by sentences?

* LangChain has a nice text splitter: https://python.langchain.com/docs/concepts/text_splitters/

### LangChain Text Splitter



### Disclaimer: --> This is an educational tool to showcase how to work with llms and RAG applications, not for commercial use. 

