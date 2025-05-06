
import os, uuid, shutil, asyncio
from ffmpeg import input as ffInput, output as ffOutput, probe
import whisper, yt_dlp, cv2, numpy as np, torch
from deep_translator import GoogleTranslator
from keybert import KeyBERT
from transformers import pipeline
from detoxify import Detoxify
from sentence_transformers import SentenceTransformer
import streamlit as st
from pyngrok import ngrok

# ─────────────────────────────────────────────────────────────────────────────
# 1) STREAMLIT PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Summarize",
    page_icon="✨",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# 2) WHITE & LAVENDER THEME
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
      /* page background lavender */
      html, body, .stApp {
        background-color: #E6E6FA !important;
      }
      /* panels & sidebar white */
      .css-1d391kg, .css-1v012nw, .stSidebar .css-ng1fyi {
        background-color: #FFFFFF !important;
      }
      /* text indigo */
      h1, h2, h3, p, label {
        color: #4B0082 !important;
      }
      /* lavender buttons with white text */
      button, .stButton>button {
        background-color: #9370DB !important;
        color: #FFFFFF !important;
        border-radius: 4px !important;
        padding: 0.5em 1em !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# 3) NAVIGATION STATE
# ─────────────────────────────────────────────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'

def go_to_main():
    st.session_state.page = 'main'

# ─────────────────────────────────────────────────────────────────────────────
# 4) WELCOME SCREEN
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.page == 'welcome':
    st.markdown("<h1 style='text-align:center;'>💖 Welcome to Summarize</h1>", unsafe_allow_html=True)
    name = st.text_input("Enter your name to begin:", key='name_input')

    # Button when name is provided
    if st.button("Start 💌", key="start_with_name") and name.strip():
        st.session_state.username = name.strip()
        go_to_main()

    # Button if user clicks without entering name
    elif st.button("Start 💌", key="start_without_name"):
        st.warning("Please tell me your name first!")

# ─────────────────────────────────────────────────────────────────────────────
# 5) MAIN SUMMARIZER SCREEN
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.page == 'main':
    st.markdown(f"### Hi {st.session_state.username}! 🎀")
    st.write("Upload a video file **or** paste a YouTube URL:")
    uploaded = st.file_uploader("Choose video file", type=['mp4','mkv','webm'])
    url = st.text_input("YouTube URL")

    if st.button("Summarize 🎬", key="summarize_button"):
        # clean up old files
        for f in os.listdir():
            if f.startswith("video_") or f.endswith((".wav", ".jpg")):
                os.remove(f)
        if os.path.isdir("scene_frames"):
            shutil.rmtree("scene_frames")

        # load models once
        device = "cuda" if torch.cuda.is_available() else "cpu"
        asyncio.set_event_loop(asyncio.new_event_loop())
        emb = SentenceTransformer('all-MiniLM-L6-v2')
        kw_model = KeyBERT(model=emb)
        whisper_model = whisper.load_model("small")
        tox = Detoxify('original')
        sent = pipeline(
            'sentiment-analysis',
            model='distilbert/distilbert-base-uncased-finetuned-sst-2-english'
        )

        # helper functions
        def download_video(url):
            uid = str(uuid.uuid4())[:8]
            opts = {'format': 'best', 'outtmpl': f'video_{uid}.%(ext)s', 'quiet': True}
            yt_dlp.YoutubeDL(opts).download([url])
            for f in os.listdir():
                if f.startswith(f"video_{uid}") and f.endswith(('.mp4', '.mkv', '.webm')):
                    return f

        def extract_audio(video_path, out="audio.wav"):
            proc = ffInput(video_path)
            ffOutput(proc, out).run(quiet=True)
            return out

        def get_duration(video_path):
            meta = probe(video_path)
            return float(meta['streams'][0]['duration'])

        def transcribe_audio(audio_path):
            return whisper_model.transcribe(audio_path)['text']

        def extract_keywords(text):
            return kw_model.extract_keywords(text, keyphrase_ngram_range=(1,2), stop_words='english', top_n=10)

        def analyze_sentiment(text):
            return sent(text[:512])[0]

        def detect_toxicity(text):
            flagged = {k:v for k,v in tox.predict(text).items() if v > 0.5}
            return flagged or "No major toxicity detected."

        def translate_text(text, lang):
            return GoogleTranslator(source='auto', target=lang).translate(text)

        def make_chapters(text, dur, n=5):
            words = text.split()
            parts = np.array_split(words, n)
            times = [round(i * dur / n, 2) for i in range(n)]
            return [{"time": times[i], "txt": " ".join(p[:15]) + "..."} for i,p in enumerate(parts)]

        def extract_scenes(video_path, thr=30.0):
            cap = cv2.VideoCapture(video_path)
            prev = None; cnt = 0
            os.makedirs("scene_frames", exist_ok=True)
            out = []
            while True:
                ok, frm = cap.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
                if prev is not None and np.abs(gray - prev).mean() > thr:
                    path = f"scene_frames/frame_{cnt}.jpg"
                    cv2.imwrite(path, frm)
                    out.append(path)
                prev = gray; cnt += 1
            cap.release()
            return out

        # Processing pipeline
        with st.spinner("Processing…"):
            # get video file
            if uploaded:
                vid_path = f"video_{uuid.uuid4().hex[:8]}.mp4"
                with open(vid_path, 'wb') as f:
                    f.write(uploaded.getbuffer())
            else:
                vid_path = download_video(url)

            audio_path = extract_audio(vid_path)
            transcript = transcribe_audio(audio_path)
            duration = get_duration(vid_path)

        # Display results
        st.subheader("📄 Transcript")
        st.write(transcript[:1000] + "..." if len(transcript) > 1000 else transcript)

        st.subheader("🔑 Keywords")
        st.write(extract_keywords(transcript))

        st.subheader("🙂 Sentiment")
        st.write(analyze_sentiment(transcript))

        st.subheader("☢ Toxicity")
        st.write(detect_toxicity(transcript))

        st.subheader("📝 Translations")
        for lang in ["en","hi","te"]:
            st.markdown(f"**{lang.upper()}**")
            st.info(translate_text(transcript, lang))

        st.subheader("🕒 Timeline")
        for chap in make_chapters(transcript, duration):
            st.markdown(f"⏱ {chap['time']}s: {chap['txt']}")

        st.subheader("🖼 Scene Frames")
        for img in extract_scenes(vid_path)[:10]:
            st.image(img, width=200)

# ─────────────────────────────────────────────────────────────────────────────
# 6) EXPOSE VIA NGROK (ONLY WHEN RUN AS SCRIPT)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # kill any old tunnels
    ngrok.kill()
    public_url = ngrok.connect(8501)
    print("👉 Streamlit URL:", public_url)
    st.write("App running at:", public_url)
