# VRAG: Multimodal Video Search Engine

**VRAG (Video Retrieval-Augmented Generation)** is a local AI application that lets users search and interact with video using natural-language questions. It combines **audio transcription, visual understanding, semantic retrieval, and LLM reasoning** into a single workflow.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LLM](https://img.shields.io/badge/LLM-Llama%203-orange)
![Vision](https://img.shields.io/badge/Vision-Moondream-purple)
![Vector%20DB](https://img.shields.io/badge/Vector%20DB-ChromaDB-green)
![UI](https://img.shields.io/badge/UI-Streamlit-red)

## Why VRAG?

Traditional video search often depends on manually scrubbing through a timeline. VRAG converts video into searchable multimodal context so users can ask questions such as:

- "Where does the red car appear?"
- "Summarize what is being said."
- "When does the screen turn blue?"

The system can return relevant context with timestamps so users can move directly to the corresponding point in the video.

## Architecture

`Video`
→ **Audio / Frames**
→ **Whisper + Moondream**
→ **Structured Multimodal Context**
→ **ChromaDB Retrieval**
→ **Llama 3**
→ **Answer + Timestamp**

## Key Features

- **Multimodal understanding** — combines spoken content and visual information.
- **Semantic video search** — retrieves relevant moments from natural-language queries.
- **Context-aware answers** — uses Llama 3 to answer against retrieved video context.
- **Timestamp navigation** — connects responses back to relevant moments in the video.
- **Local-first workflow** — uses Ollama for local model execution.

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | Llama 3 via Ollama |
| Vision | Moondream |
| Speech-to-text | Whisper |
| Vector database | ChromaDB |
| Orchestration | LangChain |
| Language | Python |

## Project Structure

```text
.
├── app.py            # Streamlit application
├── ingest.py         # Audio/visual ingestion pipeline
├── rag.py            # Retrieval and generation workflow
├── Manual.txt        # User/developer notes
├── requirements.txt
└── packages.txt
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Shikareom/VRAG-Multimodal-Engine.git
cd VRAG-Multimodal-Engine
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama models

Install [Ollama](https://ollama.com), then pull the required models:

```bash
ollama pull llama3
ollama pull moondream
```

### 4. Run the application

```bash
streamlit run app.py
```

## Usage

1. Upload an MP4 video.
2. Initialize the VRAG pipeline.
3. Allow the audio/visual ingestion process to complete.
4. Ask questions about the video in natural language.
5. Use returned timestamps to jump to relevant moments.

## Notes

The repository is intended as an experimental multimodal RAG system. Local model performance and processing time depend on the available CPU/GPU resources.

---

Built by **Om Shikare**.

[LinkedIn](https://www.linkedin.com/in/omshikare/)
