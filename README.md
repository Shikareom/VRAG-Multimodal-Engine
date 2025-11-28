# VRAG: Multimodal Video Search Engine

**VRAG (Video Retrieval-Augmented Generation)** is an AI-powered tool that understands video content. It allows users to "chat" with a video by analyzing both visual frames and audio tracks simultaneously.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![AI](https://img.shields.io/badge/AI-Ollama%20%7C%20Llama3-orange)
![Stack](https://img.shields.io/badge/Stack-Streamlit%20%7C%20ChromaDB-green)

## Features

* **1. Multimodal Intelligence:** Uses **Moondream** for visual scene understanding and **Whisper** for audio transcription.
* **2. Semantic Search:** Powered by **ChromaDB**, allowing natural language queries (e.g., *"Show me the red car"*).
* **3. Context-Aware Chat:** Integrated with **Llama 3** to answer questions based on specific video timestamps.
* **4. Smart Playback:** Click on an AI citation to instantly jump the video player to that exact second.

## Tech Stack

* **Frontend:** Streamlit (Custom UI with Chat History)
* **LLM:** Llama 3 (via Ollama)
* **Vision Model:** Moondream (via Ollama)
* **Vector DB:** ChromaDB
* **Orchestration:** LangChain / Python

## ⚙️ Installation

1.  **Clone the Repo**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/video-rag-engine.git](https://github.com/YOUR_USERNAME/video-rag-engine.git)
    cd video-rag-engine
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup Ollama**
    Download [Ollama](https://ollama.com) and pull the required models:
    ```bash
    ollama pull llama3
    ollama pull moondream
    ```

4.  **Run the App**
    ```bash
    streamlit run app.py
    ```

## Usage

1.  Upload an MP4 video file.
2.  Click **"Initialize VRAG System"**.
3.  Wait for the Audio/Visual ingestion pipeline to finish.
4.  Start chatting! (e.g., *"Summarize the speech"* or *"When does the screen turn blue?"*)

---
*Created by Om Shikare*