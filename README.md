# baseballCompanion

Local app that allows you to ask natural language questions about the current state of baseball based on recent YouTube analysis videos. It uses Whisper for transcription, SentenceTransformers for embedding, FAISS as a vector database, and llama.cpp for local LLM inference with RAG (Retrieval-Augmented Generation). Currently using Mistral 7B instruct, but flexible to model backends. No keys needed. 



![baseballCompanion](assets/logo.png)

- [baseballCompanion](#baseballcompanion)
  - [Tools:](#tools)
  - [Example - before and after RAG](#example---before-and-after-rag)
  - [Seeing Rag Results with LangChain](#seeing-rag-results-with-langchain)
  - [Example of Retrieval to Documents](#example-of-retrieval-to-documents)
  - [Prerequisites](#prerequisites)
    - [Faster Whisper](#faster-whisper)
    - [YT-DLP](#yt-dlp)
    - [LLama.cpp](#llamacpp)
    - [Embeddings](#embeddings)
      - [Question 1: Chunking](#question-1-chunking)
    - [LangChain Text Splitter](#langchain-text-splitter)
    - [How to make RAG better](#how-to-make-rag-better)
  - [Vector DB](#vector-db)
    - [Disclaimer: --\> This is an educational tool to showcase how to work with llms and RAG applications, not for commercial use.](#disclaimer----this-is-an-educational-tool-to-showcase-how-to-work-with-llms-and-rag-applications-not-for-commercial-use)



## Tools:

- **faster-whisper** for super fast transcription of videos to audio
- yt-dlp as a download engine for videos
- **llama.cpp** as an inference engine with flexible model backends and can run on apple silicon
- **mistral-7b-instruct** as the base model
- **sentence_transformers** as the sentence embedding model
- **LangChain** for RAG, query construction, query refinement, etc..




## Example - before and after RAG

The date is May 15, 2025. There are several trade rumors about Luis Robert to the Mets. Let's ask our assistant what people are saying about this:

**question:** 'what are people saying today about trade rumors of Luis Robert to the Mets?'

Pre-context answer:

```
I don't have real-time information about specific trade rumors, but I can tell you that Luis Robert is a promising young outfielder who has been playing for the Chicago White Sox. He is known for his impressive performance at the plate and his speed on the bases. Some people may be interested in knowing what the Mets are thinking about the possibility of acquiring him, but it is important to note that nothing is certain until a trade is officially announced. Is there anything else you would like to know?<|im_end|>
```

Post-context answer:

```
Some people may be excited about the possibility of having him on the team, as he is known for his impressive performance at the plate and his speed on the bases. However, it is important to note that nothing is certain until a trade is officially announced. 

The video and podcast episode you mentioned also provide some context about Luis Robert's performance in recent seasons, including his batting average, 
on-base percentage, slugging, and OPS. It also mentions that he has been playing bad for two years, but that it is hard not to see what he did in previous seasons. 

In terms of value, a trade for Luis Robert would likely involve giving up a significant amount of assets, such as top prospects 
or established players. It is also worth considering the potential risk of injury, as Luis Robert has only played more than 101 games in one season in his entire career. 

Is there anything else you would like to know about this topic?
```

## Seeing Rag Results with LangChain

* This is using top 3 retrieval for context into a non-finetuned Mistral7B: run `chat.py` to try. 


```
Prompt after formatting:
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Luis Robert, trade rumors are real. Bob Nightingale has reported that Luis Robert Jr. and the New York Mets have been linked in possible trades. The White Sox are looking at Mets prospects. We're gonna talk about everything that there is to possibly talk about with these Luis Robert trade rumors in today's YouTube video and podcast episode. Make sure you are subscribed to the Mets on Podcast, YouTube channel, so don't miss out on any of the content coming at you, videos after every single series, and a third bonus episode every single week. So you're gonna want to stick around and see that. And if you are listening to us on Apple, podcast, Spotify, Google, whatever it is, drop us a reading, drop us a review, download and subscribe. We really do appreciate it. James, when I saw this news, I know we've talked about this off camera many a times. Luis Robert would be awesome on this team. He'd be fantastic to have. He'd be kind of perfect, because when you look up and down this Mets

lineup, it's very clearly center field. Luis Robert Jr.'s that spot right now where he has two years of team options left on his White Sox contract, $20 million each for the next two seasons and both totally voluntary money. So the White Sox can just get rid of him from the end of the season if they don't want to pay. If you trade for him, I'm sure it won't happen. If you want to get rid of him, you could do that with no risk at all. And he has been playing bad for two years, but I think we all know the potential a guy like him has. And that was funny for me to see during our last video when we talked about it briefly, how much the Mets fan don't like this idea, how much they don't want Luis Robert, how much he sullied his name in baseball circles over his last 150 games when he's been really, really bad, but kind of forgetting the four years before that when he was really, really good the entire time. Yeah, I mean, let's give the people some context in case they don't know. Over his

by again, the other comment, I think that was by MacDog and then also by Ken Rosendahl in South Territory is that this, because Will Sam and also somewhat perpude this too and report on the athletics saying like, it's not a lie that the Mets have checked in Luis Robert, have been somewhat connected to the White Sox and Robert with the center field being the spot of relative weakness in their lineup, but also being acutely aware that this is not the area of the most dire need. Cause the Mets do drop a hammer at some point, this trade deadline, it will likely be for a starting pitcher. And we still don't know where those dominoes will fall. We don't know what teams are going to be selling yet. It's way too early to know that, but being over the fourth threshold of the luxury tax, the Cohen tax, we're paying 110% tax on every single dollar that we spend above that. So if you bring in $20 million Luis Robert contract, that is $42 million. For the rest of the season, that'll be $20 million

Question: hello, I am wondering if you can give me a run down on whats going on with the Luis Robert going to the Mets rumors?
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

And the above chunk would clearly be a very good context chunk for an LLM to have when asked about the current state of the New York Mets for example. 


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

### How to make RAG better

* There _is_ an optimal chunk size

* Add metadata into chunks

    * This makes it so that we can add 'structured data'-like filters onto our unstructured data (podcast transcriptions)

* Embed other things like 'what questions can this chunk answer'

* Small-to-big retrieval

    * when we embed an entire chunk of 20-30 sentences of content, there might be some fluff around it. We cant embed every sentence though since that could make things slow, or miss context in retrieval. A nice middle ground is to try small-to-big retrieval. 

## Vector DB



### Disclaimer: --> This is an educational tool to showcase how to work with llms and RAG applications, not for commercial use. 

