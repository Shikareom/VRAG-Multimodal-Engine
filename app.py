import streamlit as st
import os
import ollama
import json

#importing backend logic
try:
    from ingest import extract_and_transcribe_audio, analyze_visual_frames
    from rag import index_data, search_video
except ImportError:
    st.error("Error: Could not find 'ingest.py' or 'rag.py'.")
    st.stop()

# PAGE CONFIG
st.set_page_config(layout="wide", page_title="VRAG Engine", initial_sidebar_state="collapsed")

# CSS - DESIGN & FOOTER
st.markdown("""
<style>
    /* Force White Background */
    .stApp {
        background-color: #080e13;
    }
    /* Center the File Uploader */
    [data-testid="stFileUploader"] {
        width: 100%;
        margin: 0 auto;
    }
    /* Style the Process Button */
    .stButton button {
        width: 100%;
        background-color: #000000;
        color: white;
        border-radius: 8px;
        font-weight: bold;
    }
    /* Hide the top header decoration */
    header {visibility: hidden;}
    
    /* FOOTER STYLING */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #ffffff;
        color: #888;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-top: 1px solid #eee;
        z-index: 1000;
    }
</style>
""", unsafe_allow_html=True)

# SESSION STATE
if 'timestamp' not in st.session_state:
    st.session_state.timestamp = 0
if 'history' not in st.session_state:
    st.session_state.history = []
if 'processed' not in st.session_state:
    st.session_state.processed = False

# CENTERED UPLOAD SECTION
st.markdown("<h1 style='text-align: center; color: #fff;'>VRAG: Multimodal Search Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Upload a video to chat with its visual and audio content.</p>", unsafe_allow_html=True)

spacer1, center_col, spacer2 = st.columns([1, 2, 1])

uploaded_file = None
video_path = "data/uploaded_video.mp4"

with center_col:
    uploaded_file = st.file_uploader("Upload MP4", type=['mp4'], label_visibility="collapsed")
    
    if uploaded_file:
        if not os.path.exists("data"):
            os.makedirs("data")
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("Initialize Analysis"):
            with st.spinner("Ingesting Video & Audio Data..."):
                audio_docs = extract_and_transcribe_audio(video_path)
                visual_docs = analyze_visual_frames(video_path)
                
                full_index = audio_docs + visual_docs
                with open("data/video_index.json", "w") as f:
                    json.dump(full_index, f)
                
                index_data()
                st.session_state.processed = True
                st.rerun()

st.divider()

#  MAIN DISPLAY 
if st.session_state.processed and uploaded_file:
    
    vid_col, chat_col = st.columns([1, 2])
    
    with vid_col:
        st.subheader("Source Feed")
        st.video(uploaded_file, start_time=int(st.session_state.timestamp))
        st.caption(f"⏱ Current Timestamp: {st.session_state.timestamp}s")

    with chat_col:
        st.subheader("VRAG Assistant")
        chat_container = st.container(height=500)
        
        with chat_container:
            if not st.session_state.history:
                st.info("System Ready. Ask questions like 'What is the person holding?'")
            
            for chat in st.session_state.history:
                with st.chat_message("user", avatar="🔘"):
                    st.write(chat['question'])
                with st.chat_message("assistant", avatar="🟢"):
                    st.write(chat['answer'])
                    if st.button(f"▶ Jump to {chat['time']}s", key=f"btn_{chat['id']}"):
                        st.session_state.timestamp = chat['time']
                        st.rerun()

        query = st.chat_input("Query video content...")
        
        if query:
            results = search_video(query, top_k=3)
            
            if results:
                top_result = results[0]
                time_sec = int(top_result['start'])
                
                prompt = f"""
                Context: "{top_result['content']}"
                Question: "{query}"
                Answer concisley based on context.
                """
                try:
                    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
                    ai_answer = response['message']['content']
                except:
                    ai_answer = "Sorry, I couldn't generate an answer."
                
                st.session_state.history.append({
                    "id": len(st.session_state.history),
                    "question": query,
                    "answer": ai_answer,
                    "time": time_sec
                })
                st.session_state.timestamp = time_sec
                st.rerun()
            else:
                st.warning("No answer found in video.")

elif uploaded_file and not st.session_state.processed:
    st.info("Click 'Initialize' to activate the AI.")

# footer
st.markdown("""
<div class="footer">
    ©omshikare7077@gmail.com 
</div>
""", unsafe_allow_html=True)
