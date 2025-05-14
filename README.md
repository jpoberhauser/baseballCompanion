# baseballCompanion

## Execution and Plan

Setup & Proof of Concept 
	•	Set up llama.cpp locally.
	•	Run Whisper on a test video and store transcript.
	•	Build embedding + vector store prototype.
	•	Manual prompt + RAG integration with llama.cpp.

Backend Pipeline 
	•	Automate YouTube → transcript → embedding → store.
	•	Script for refreshing content weekly/daily.
	•	Test with multiple baseball YouTube episodes.

Desktop App 
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