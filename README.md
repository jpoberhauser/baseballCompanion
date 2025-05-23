# baseballCompanion

Local app that allows you to ask natural language questions about the current state of baseball based on recent YouTube analysis videos. It uses Whisper for transcription, SentenceTransformers for embedding, FAISS as a vector database, and llama.cpp for local LLM inference with RAG (Retrieval-Augmented Generation). Currently using Mistral 7B instruct, but flexible to model backends. No keys needed. 



![baseballCompanion](assets/logo.png)

- [baseballCompanion](#baseballcompanion)
  - [Tools:](#tools)
  - [Example - before and after RAG](#example---before-and-after-rag)
    - [Before RAG:](#before-rag)
    - [After RAG:](#after-rag)
  - [Relevant Context being Passed](#relevant-context-being-passed)
  - [Source 1:](#source-1)
  - [Source 2:](#source-2)
  - [Source 3:](#source-3)
- [Content: this? Who's telling him this? That's the other thing. Cause there's now a rat allegedly. There's a rat somewhere, somewhere in the line. There is a rat right now. But that's official. There's a rat. The only time Soto seemed happy this weekend was while he interacted with judge manager Aaron Boone. As I said, I'm no mind reader. Again, doubling down the fact that he has no idea what he's talking about. But my hunch is that not only does Soto miss the Yankees, but the permanence of his decision is sinking in. Soto is staring at another 14 plus years and flushing. Now, just for some context about the permanence that is setting in for Juan Soto and the sinking feeling he has for his decision, settling in, just to keep in mind, the Mets are currently 29 and 19. The Yankees are 27 and 19. Both some of the best teams in Major League Baseball. Juan Soto is getting paid $750 million, $50 million a year. I don't think there is any sort of grief setting in of Juan Soto of, oh no, I made a](#content-this-whos-telling-him-this-thats-the-other-thing-cause-theres-now-a-rat-allegedly-theres-a-rat-somewhere-somewhere-in-the-line-there-is-a-rat-right-now-but-thats-official-theres-a-rat-the-only-time-soto-seemed-happy-this-weekend-was-while-he-interacted-with-judge-manager-aaron-boone-as-i-said-im-no-mind-reader-again-doubling-down-the-fact-that-he-has-no-idea-what-hes-talking-about-but-my-hunch-is-that-not-only-does-soto-miss-the-yankees-but-the-permanence-of-his-decision-is-sinking-in-soto-is-staring-at-another-14-plus-years-and-flushing-now-just-for-some-context-about-the-permanence-that-is-setting-in-for-juan-soto-and-the-sinking-feeling-he-has-for-his-decision-settling-in-just-to-keep-in-mind-the-mets-are-currently-29-and-19-the-yankees-are-27-and-19-both-some-of-the-best-teams-in-major-league-baseball-juan-soto-is-getting-paid-750-million-50-million-a-year-i-dont-think-there-is-any-sort-of-grief-setting-in-of-juan-soto-of-oh-no-i-made-a)
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


### Before RAG:


**question**

**What is the situation with the Juan Soto and the Mets in 2025? please answer in bullet points**

**answer**

Assistant: - As of October 2023, Juan Soto is under contract with the San Diego Padres through the 2024 season.
- There have been speculations regarding his potential trade or free agency, but no definitive moves have been made for him to join the New York Mets by 2025.
- The Mets have shown interest in acquiring star players, which could include Soto if he becomes available.
- Soto's performance and the Padres' success in 2024 will likely influence his future, including any potential dealings with the Mets.
- By 2025, the Mets could be in a position to pursue Soto if his contract situation changes or if he becomes a free agent after the 2024 season.


### After RAG:

**question**


**What is the situation with the Juan Soto and the Mets in 2025? please answer in bullet points**

**answer**

Assistant: - Juan Soto is under a massive contract worth $750 million, with an annual salary of $50 million.
- As of 2025, there are discussions about Soto’s performance and potential struggles, indicating he may not be meeting his own standards.
- The Mets are performing well in the league, currently holding a record of 29-19, suggesting they are competitive.
- There is speculation about Soto’s feelings towards his decision to join the Mets, with hints that he may miss the Yankees and the permanence of his contract is sinking in.
- Media narratives are focusing on Soto’s competitiveness and the pressure he feels to live up to his contract.
- The situation is being closely observed by fans and journalists, with some expressing dissatisfaction with the coverage and narratives surrounding Soto’s performance.


## Relevant Context being Passed

* This is using top 3 retrieval for context run `chat_rag.py` to try. 

**question**

**What is the situation with the Juan Soto and the Mets in 2025? please answer in bullet points**

**context**

Source 1:
----------------------------------------
Content: about? But then it does come back to what we were alluding to that Juan Soto knows he's not playing up to his own standards. It's painfully obvious. I don't think anybody on earth is more aware of it than him, despite with every single Mets Yankee fan journalists on earth who want you to believe that they're the ones who are the most aware of it. No one cares right now more about Juan Soto's struggles than Juan Soto. The guy is competitive as fuck. He's one of the best players of all time and there's a lot of pressure from to live up to this massive contract and he intends to live up to it. Nothing there has changed. It's been seven weeks of baseball. People are like, I can't believe this is happening. Oh my God, he's so unhappy. Again, these are Mets fans that have not been cut from the same cloth. These are Mets fans that do not remember 2005 Carlos Beltran. They don't even remember first half 2022 Francisco Lindor, remember that? We were a year and a half into this shit and people

Source 2:
----------------------------------------
Content: that one, there's, yeah, and this podcast, there's insanely bad journalism going on at times where it's just like the stupidest articles, the dumbest quotes. I mean, we do it to Mets people. So this isn't just a Mets Yankee thing. We'll media marvel anybody. We've media marveled ourselves on this podcast. Multiple times. So we will go after anybody that says something stupid. But the sense of entitlement that these guys have, I'll tell you this, like, I think from a vice perspective, I definitely don't love the idea of Juan Soto just leaving. Like that, I think there is something a little bit weird there. But for them to be like so sour grapes, because they couldn't get the quote that they were trying to set Juan Soto up for, because Mike Puma kind of opened the door with this a couple of weeks ago when he got the air and judge quote. It ties back to that a lot. Everyone's like, this fucking Mike Puma guy gets this quote. How the hell does he get it? That could have been mine.

Source 3:
----------------------------------------
Content: this? Who's telling him this? That's the other thing. Cause there's now a rat allegedly. There's a rat somewhere, somewhere in the line. There is a rat right now. But that's official. There's a rat. The only time Soto seemed happy this weekend was while he interacted with judge manager Aaron Boone. As I said, I'm no mind reader. Again, doubling down the fact that he has no idea what he's talking about. But my hunch is that not only does Soto miss the Yankees, but the permanence of his decision is sinking in. Soto is staring at another 14 plus years and flushing. Now, just for some context about the permanence that is setting in for Juan Soto and the sinking feeling he has for his decision, settling in, just to keep in mind, the Mets are currently 29 and 19. The Yankees are 27 and 19. Both some of the best teams in Major League Baseball. Juan Soto is getting paid $750 million, $50 million a year. I don't think there is any sort of grief setting in of Juan Soto of, oh no, I made a
============================================================



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

