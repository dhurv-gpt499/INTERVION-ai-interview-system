import warnings
warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_ENTITY.*")
warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_CONTENT.*")
warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")
warnings.filterwarnings("ignore", message=".*theme.*Blocks constructor.*")

import gradio as gr
import threading
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_processor.pipeline import run_pipeline
from audio_processor import tts_engine
from resume_parser.resume_parser import parse_resume

# ── Custom Premium CSS ───────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

body, .gradio-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%) !important;
    font-family: 'Inter', sans-serif !important;
    color: #f8fafc !important;
}

/* Glassmorphism Panels */
.glass-panel {
    background: rgba(30, 41, 59, 0.7) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5) !important;
    padding: 24px !important;
    margin-bottom: 20px !important;
}

/* Text & Headers */
h1, h2, h3, h4, h5 {
    color: #f8fafc !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 800 !important;
}

.hero-title {
    text-align: center;
    background: linear-gradient(90deg, #60a5fa, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3.5rem !important;
    margin-bottom: 0px !important;
    padding-bottom: 10px !important;
}
.hero-subtitle {
    text-align: center;
    color: #cbd5e1 !important;
    font-size: 1.2rem !important;
    margin-top: 5px !important;
    margin-bottom: 30px !important;
}

/* Buttons */
.primary-btn {
    background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 800 !important;
    font-size: 1.2rem !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
}
.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(168, 85, 247, 0.6) !important;
}

.stop-btn {
    background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 800 !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}
.stop-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(239, 68, 68, 0.5) !important;
}

/* Avatar Styling */
#avatar_col img {
    border-radius: 20px !important;
    box-shadow: 0 0 50px rgba(99, 102, 241, 0.4) !important;
    border: 2px solid rgba(255, 255, 255, 0.1) !important;
    background: #0f172a !important;
    transition: box-shadow 0.3s ease !important;
}

/* Webcam Styling */
#webcam_col img {
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}
"""

# ── Avatar animated GIFs ──────────────────────────────────────
ASSETS = os.path.join(os.path.dirname(__file__), "assets")

AVATAR_GIFS = {
    "talking"  : "talking.webp",
    "idle"     : "idle.webp",
    "listening": "listening.webp",
    "thinking" : "thinking.webp",
}

# ── Shared state (pipeline writes, UI reads) ──────────────────────────
shared = {
    "screen"      : "setup", # "setup", "loading", "interview", "report"
    "transcript"  : "",
    "state"       : "Waiting to start...",
    "avatar"      : "idle",
    "anxiety"     : 0.0,
    "confidence"  : 100.0,
    "webcam"      : None,
    "qa_history"  : [],
    "running"     : False,
    "frame_tick"  : 0,
    "report_card" : {},
}

pipeline_thread = None

# ── Pipeline callbacks ────────────────────────────────────────────────
def on_state_change(state_name: str):
    shared["state"] = state_name
    if state_name in ("ai_speaking",):
        shared["avatar"] = "talking"
        shared["frame_tick"] = 0
    elif state_name in ("listening", "candidate_paused"):
        shared["avatar"] = "listening"
    elif state_name in ("evaluating",):
        shared["avatar"] = "thinking"
    else:
        shared["avatar"] = "idle"

def on_transcript_update(text: str):
    shared["transcript"] += f"\n{text}"

def on_qa_complete(question: str, answer: str):
    shared["qa_history"].append({"q": question, "a": answer})

def on_vision_scores(anxiety: float, confidence: float):
    shared["anxiety"] = anxiety
    shared["confidence"] = confidence

def on_vision_frame(frame):
    try:
        import cv2
        frame = cv2.resize(frame, (320, 240))
    except Exception:
        pass
    shared["webcam"] = frame


# ── Start interview ────────────────────────────────────────────────────
def start_interview(resume_file, companies_str, roles_str, level, duration, focus_weak, mic_device_str):
    global pipeline_thread

    if shared["running"]:
        return "Already running!"

    if resume_file is None:
        return "Please upload a resume first."

    file_path = resume_file.name if hasattr(resume_file, "name") else str(resume_file)

    if isinstance(companies_str, list):
        companies = [c.strip() for c in companies_str if c.strip()]
    else:
        companies = [c.strip() for c in str(companies_str).split(",") if c.strip()]
    roles     = [r.strip() for r in roles_str.split(",") if r.strip()]

    # reset shared state
    shared["transcript"] = ""
    shared["state"]      = "Parsing Resume — please wait..."
    shared["avatar"]     = "idle"
    shared["qa_history"] = []
    shared["running"]    = True
    shared["screen"]     = "interview"

    mic_index = None
    if mic_device_str:
        try:
            mic_index = int(mic_device_str.split(":")[0])
        except Exception:
            pass

    # run pipeline in background thread
    pipeline_thread = threading.Thread(
        target      = _run_pipeline_thread,
        args        = (file_path, companies, roles, level, int(duration), bool(focus_weak), mic_index),
        daemon      = True,
    )
    pipeline_thread.start()
    return "Starting..."


def _run_pipeline_thread(resume_file_path, companies, roles, level, duration, focus_weak=True):
    resume_parsed = {}
    try:
        resume_parsed, _ = parse_resume(resume_file_path)
        
        shared["state"] = "Loading AI Model (takes ~10s)..."
        run_pipeline(
            resume_parsed       = resume_parsed,
            preferred_companies = companies,
            preferred_roles     = roles,
            target_level        = level,
            duration_minutes    = duration,
            focus_weaknesses    = focus_weak,
            on_state_change     = on_state_change,
            on_transcript       = on_transcript_update,
            on_qa_complete      = on_qa_complete,
            on_vision_scores    = on_vision_scores,
            on_vision_frame     = on_vision_frame,
            is_running          = lambda: shared["running"],
        )
    finally:
        shared["running"] = False
        shared["state"]   = "Generating Report Card..."
        shared["avatar"]  = "idle"
        
        try:
            from llm_interviewer.answer_evaluator import AnswerEvaluator
            from database.database import save_interview_report
            import uuid

            evaluator = AnswerEvaluator()
            report = evaluator.evaluate_final_interview(
                qa_history          = shared["qa_history"],
                resume_parsed       = resume_parsed if isinstance(resume_parsed, dict) else {},
                preferred_companies = companies,
                target_level        = level,
                avg_anxiety         = shared["anxiety"],
                avg_confidence      = shared["confidence"]
            )
            shared["report_card"] = report
            
            session_id = str(uuid.uuid4())[:8]
            save_interview_report(
                session_id       = session_id,
                candidate_name   = resume_parsed.get("name", "Candidate") if isinstance(resume_parsed, dict) else "Candidate",
                target_companies = companies,
                target_roles     = roles,
                target_level     = level,
                overall_score    = report.get("overall_score", 70),
                verdict          = report.get("verdict", "Complete"),
                report_json      = report
            )
        except Exception as e:
            print(f"[REPORT ERROR] {e}")
            shared["report_card"] = {}

        shared["screen"] = "report"


def stop_interview():
    shared["running"] = False
    tts_engine.stop()
    shared["state"]   = "Interview ended. Generating Report Card..."
    return "Stopped."


# ── Shared state for caching poll values ───────────────────────────────
_last_avatar_state = None
_last_webcam_frame = None

def poll():
    global _last_avatar_state, _last_webcam_frame

    show_setup     = gr.update(visible=(shared["screen"] == "setup"))
    show_loading   = gr.update(visible=False)
    show_interview = gr.update(visible=(shared["screen"] == "interview"))
    show_report    = gr.update(visible=(shared["screen"] == "report"))

    state = shared["avatar"]
    if state != _last_avatar_state:
        avatar_img = os.path.join(ASSETS, AVATAR_GIFS.get(state, "idle.webp"))
        _last_avatar_state = state
    else:
        avatar_img = gr.skip()

    history_md = "\n\n".join(
        f"**Q{i+1}:** {qa['q']}\n\n**A:** {qa['a']}"
        for i, qa in enumerate(shared["qa_history"])
    ) or "No answers yet."

    rc = shared.get("report_card", {})
    score_str      = f"🏆 {rc.get('overall_score', 0)} / 100" if rc else "Pending..."
    verdict_str    = str(rc.get("verdict", "N/A")) if rc else "Pending..."
    summary_str    = str(rc.get("summary", "No evaluation generated yet.")) if rc else ""
    strengths_str  = "\n".join(f"🟢 {s}" for s in rc.get("strengths", [])) if rc and rc.get("strengths") else "None recorded."
    weaknesses_str = "\n".join(f"🔴 {w}" for w in rc.get("weaknesses", [])) if rc and rc.get("weaknesses") else "None recorded."
    
    topics_dict    = rc.get("topic_scores", {}) if rc else {}
    topics_str     = "\n".join(f"• {k}: **{v}/100**" for k, v in topics_dict.items()) if isinstance(topics_dict, dict) else str(topics_dict)
    rec_str        = str(rc.get("key_recommendation", "Continue practicing technical fundamentals.")) if rc else ""

    if shared["webcam"] is not _last_webcam_frame:
        webcam_img = shared["webcam"]
        _last_webcam_frame = shared["webcam"]
    else:
        webcam_img = gr.skip()

    return (
        show_setup,
        show_loading,
        show_interview,
        show_report,
        avatar_img,
        shared["state"],
        shared["transcript"].strip(),
        shared["anxiety"],
        shared["confidence"],
        webcam_img,
        history_md,
        score_str,
        verdict_str,
        summary_str,
        strengths_str,
        weaknesses_str,
        topics_str,
        rec_str,
    )


COMPANY_CHOICES = [
    "DE Shaw", "Citadel", "Two Sigma", "Jane Street", "Jump Trading", 
    "Tower Research Capital", "Arcesium", "Bloomberg", "WorldQuant", "Trexquant",
    "Goldman Sachs", "Morgan Stanley", "JP Morgan Chase", "Barclays", "Visa", "Mastercard", "PayPal", "Stripe", "Square / Block", "Intuit",
    "Google", "Meta", "Apple", "Amazon", "Microsoft", "Netflix", "NVIDIA", "Adobe", "Salesforce", "Uber",
    "Oracle", "Cloudflare", "Cisco Systems", "Palo Alto Networks", "Datadog", "Snowflake", "MongoDB", "VMware", "Red Hat", "Akamai Technologies",
    "Flipkart", "Zomato", "Swiggy", "Razorpay", "Atlassian", "LinkedIn", "Airbnb", "Expedia", "McKinsey / Tech Consulting", "Boston Consulting Group (BCG)"
]

# ── Gradio UI ─────────────────────────────────────────────────────────
from audio_processor.mic_capture import AudioCapture
try:
    devices_info = AudioCapture.get_devices()
    mic_choices = [f"{d['index']}: {d['name']}" for d in devices_info]
    default_mic = next((f"{d['index']}: {d['name']}" for d in devices_info if d.get("is_default")), None)
    if not default_mic and mic_choices:
        default_mic = mic_choices[0]
except Exception:
    mic_choices = []
    default_mic = None

theme = gr.themes.Base(
    primary_hue="indigo",
    secondary_hue="purple",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)

with gr.Blocks(theme=theme, css=CUSTOM_CSS, title="INTERVION") as app:
    
    # Hero Section
    gr.HTML("<h1 class='hero-title'>INTERVION</h1>")
    gr.HTML("<p class='hero-subtitle'>State-of-the-art AI Mock Interviewer powered by Whisper & Qwen 2.5</p>")

    # ── SCREEN 1: SETUP ───────────────────────────────────────────
    with gr.Column(visible=True, elem_classes=["setup-container"]) as setup_screen:
        
        with gr.Row():
            # Card 1: Candidate Profile
            with gr.Column(elem_classes=["glass-panel"]):
                gr.Markdown("### 👤 Candidate Profile")
                resume_file  = gr.File(label="Upload Resume (PDF)", type="filepath")
                level        = gr.Dropdown(
                                choices=["intern", "entry", "mid", "senior"],
                                value="entry",
                                label="Experience Level"
                               )

            # Card 2: Target & Strategy
            with gr.Column(elem_classes=["glass-panel"]):
                gr.Markdown("### 🎯 Interview Strategy")
                companies    = gr.Dropdown(
                                choices=COMPANY_CHOICES,
                                value=["Jane Street", "Google"],
                                multiselect=True,
                                label="Target Companies (Determines AI Persona)"
                               )
                roles        = gr.Textbox(label="Target Roles", placeholder="e.g., Software Engineer, Quant Researcher")
                duration     = gr.Slider(10, 60, value=20, step=5, label="Duration (minutes)")
                focus_weak   = gr.Checkbox(value=True, label="Focus on weaknesses from past sessions")
                mic_dropdown = gr.Dropdown(
                                choices=mic_choices,
                                value=default_mic,
                                label="🎤 Select Microphone"
                               )
        
        with gr.Row():
            start_btn = gr.Button("Initialize Interview", elem_classes=["primary-btn"], size="lg")
        status_box = gr.Textbox(label="Status", interactive=False, value="Ready", visible=False)

    # ── SCREEN 2: LOADING ─────────────────────────────────────────
    with gr.Column(visible=False, elem_classes=["glass-panel"]) as loading_screen:
        gr.Markdown("<br><br><h2 style='text-align: center;'>⏳ Securing Secure Connection & Loading AI Weights...</h2>")
        gr.Markdown("<p style='text-align: center;'>Please wait ~10 seconds. The interview will start automatically.</p>")

    # ── SCREEN 3: INTERVIEW ───────────────────────────────────────
    with gr.Column(visible=False) as interview_screen:
        with gr.Row():
            # Left panel - Stats and controls
            with gr.Column(scale=1):
                with gr.Column(elem_classes=["glass-panel"]):
                    state_box = gr.Textbox(
                                  label="System Status",
                                  value="Waiting...",
                                  interactive=False,
                                )
                    gr.Markdown("<br>")
                    webcam_img = gr.Image(label="Live Feed", interactive=False, height=180, elem_id="webcam_col")
                
                with gr.Accordion("Advanced Vision Analytics", open=False, elem_classes=["glass-panel"]):
                    anxiety_slider    = gr.Slider(0, 100, value=0, label="😰 Anxiety Metric",    interactive=False)
                    confidence_slider = gr.Slider(0, 100, value=0, label="💪 Confidence Metric", interactive=False)
                
                gr.Markdown("<br>")
                stop_btn  = gr.Button("End Session", elem_classes=["stop-btn"])

            # Center - Avatar
            with gr.Column(scale=2, elem_id="avatar_col"):
                avatar_img  = gr.Image(
                                value=os.path.join(ASSETS, AVATAR_GIFS["idle"]),
                                interactive=False,
                                show_label=False
                              )

        # Bottom - collapsible transcript/history
        with gr.Column(elem_classes=["glass-panel"]):
            with gr.Accordion("📝 Live AI Transcript", open=True):
                transcript_box = gr.Textbox(
                                   show_label=False,
                                   lines=6,
                                   interactive=False,
                                 )
            with gr.Accordion("📋 Full Q&A History", open=False):
                history_md = gr.Markdown("No answers yet.")

    # ── SCREEN 4: REPORT CARD ─────────────────────────────────────
    with gr.Column(visible=False, elem_classes=["glass-panel"]) as report_screen:
        gr.Markdown("<h2 style='text-align: center; color: #a855f7 !important;'>🏆 Final Assessment Report</h2>")
        gr.Markdown("<p style='text-align: center;'>Generated by INTERVION AI Hiring Committee</p><br>")
        
        with gr.Row():
            with gr.Column(scale=1):
                score_box   = gr.Textbox(label="Overall Score", value="0 / 100", interactive=False)
                verdict_box = gr.Textbox(label="Hiring Verdict", value="Pending...", interactive=False)
            with gr.Column(scale=2):
                summary_box = gr.Textbox(label="Executive Summary", lines=4, interactive=False)
        
        with gr.Row():
            with gr.Column():
                strengths_box  = gr.Textbox(label="Demonstrated Strengths", lines=6, interactive=False)
            with gr.Column():
                weaknesses_box = gr.Textbox(label="Critical Areas for Improvement", lines=6, interactive=False)

        with gr.Row():
            with gr.Column(scale=1):
                topic_scores_box   = gr.Textbox(label="Competency Breakdown", lines=5, interactive=False)
            with gr.Column(scale=1):
                recommendation_box = gr.Textbox(label="Key Recommendation", lines=5, interactive=False)
        
        with gr.Row():
            restart_btn = gr.Button("Return to Dashboard", elem_classes=["primary-btn"], size="lg")

    # ── Timer — polls shared state every second ────────────────────────
    timer = gr.Timer(value=0.1)
    timer.tick(
        fn      = poll,
        outputs = [setup_screen, loading_screen, interview_screen, report_screen,
                   avatar_img, state_box, transcript_box,
                   anxiety_slider, confidence_slider, webcam_img, history_md,
                   score_box, verdict_box, summary_box, strengths_box, weaknesses_box, topic_scores_box, recommendation_box],
    )

    # ── Button events ──────────────────────────────────────────────────
    start_btn.click(
        fn      = start_interview,
        inputs  = [resume_file, companies, roles, level, duration, focus_weak, mic_dropdown],
        outputs = [status_box],
    )

    stop_btn.click(
        fn      = stop_interview,
        outputs = [status_box],
    )

    def restart_to_setup():
        shared["screen"]      = "setup"
        shared["report_card"] = {}
        shared["qa_history"]  = []
        shared["transcript"]  = ""
        shared["state"]       = "Ready"
        return "Ready"

    restart_btn.click(
        fn      = restart_to_setup,
        outputs = [status_box],
    )


if __name__ == "__main__":
    app.launch(share=False)
