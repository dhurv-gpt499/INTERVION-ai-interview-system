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

# ── Avatar animated GIFs ──────────────────────────────────────
ASSETS = os.path.join(os.path.dirname(__file__), "assets")

AVATAR_GIFS = {
    "talking"  : "talking.gif",
    "idle"     : "idle.gif",
    "listening": "listening.gif",
    "thinking" : "thinking.gif",
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
    "frame_tick"  : 0,   # increments every poll tick for animation
    "report_card" : {},  # stores final evaluation JSON
}

pipeline_thread = None

# ── Pipeline callbacks (called from pipeline thread) ──────────────────
def on_state_change(state_name: str):
    shared["state"] = state_name
    if state_name in ("ai_speaking",):
        shared["avatar"] = "talking"
        shared["frame_tick"] = 0   # reset so mouth starts from frame 0
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
        # Resize to drastically reduce network payload and UI stutter
        frame = cv2.resize(frame, (320, 240))
    except Exception:
        pass
    shared["webcam"] = frame


# ── Start interview ────────────────────────────────────────────────────
def start_interview(resume_file, companies_str, roles_str, level, duration, focus_weak=True):
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
    shared["screen"]     = "interview"  # go straight to interview screen

    # run pipeline in background thread
    pipeline_thread = threading.Thread(
        target      = _run_pipeline_thread,
        args        = (file_path, companies, roles, level, int(duration), bool(focus_weak)),
        daemon      = True,
    )
    pipeline_thread.start()
    return "Starting..."


def _run_pipeline_thread(resume_file_path, companies, roles, level, duration, focus_weak=True):
    resume_parsed = {}
    try:
        # Move heavy parsing to background thread so UI doesn't block!
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
    tts_engine.stop()          # ← kill audio immediately, flush queues
    shared["state"]   = "Interview ended. Generating Report Card..."
    return "Stopped."


# ── Shared state for caching poll values to prevent UI lag ─────────────
_last_poll = {}

# ── Polling — updates UI every 0.1s ────────────────────────────────────
def poll():
    show_setup     = gr.update(visible=(shared["screen"] == "setup"))
    show_loading   = gr.update(visible=False)
    show_interview = gr.update(visible=(shared["screen"] == "interview"))
    show_report    = gr.update(visible=(shared["screen"] == "report"))

    state    = shared["avatar"]
    avatar_img = os.path.join(ASSETS, AVATAR_GIFS.get(state, "idle.gif"))

    history_md = "\n\n".join(
        f"**Q{i+1}:** {qa['q']}\n\n**A:** {qa['a']}"
        for i, qa in enumerate(shared["qa_history"])
    ) or "No answers yet."

    rc = shared.get("report_card", {})
    score_str      = f"{rc.get('overall_score', 0)} / 100" if rc else "Pending..."
    verdict_str    = str(rc.get("verdict", "N/A")) if rc else "Pending..."
    summary_str    = str(rc.get("summary", "No evaluation generated yet.")) if rc else ""
    strengths_str  = "\n".join(f"• {s}" for s in rc.get("strengths", [])) if rc and rc.get("strengths") else "None recorded."
    weaknesses_str = "\n".join(f"• {w}" for w in rc.get("weaknesses", [])) if rc and rc.get("weaknesses") else "None recorded."
    
    topics_dict    = rc.get("topic_scores", {}) if rc else {}
    topics_str     = "\n".join(f"• {k}: {v}/100" for k, v in topics_dict.items()) if isinstance(topics_dict, dict) else str(topics_dict)
    rec_str        = str(rc.get("key_recommendation", "Continue practicing technical fundamentals.")) if rc else ""

    current_values = (
        show_setup,
        show_loading,
        show_interview,
        show_report,
        avatar_img,
        shared["state"],
        shared["transcript"].strip(),
        shared["anxiety"],
        shared["confidence"],
        shared["webcam"],
        history_md,
        score_str,
        verdict_str,
        summary_str,
        strengths_str,
        weaknesses_str,
        topics_str,
        rec_str,
    )

    returns = []
    for i, val in enumerate(current_values):
        # We skip updates if the value hasn't changed.
        # For the webcam image (numpy array), we check object identity (is) 
        # because a new array is assigned each frame.
        if i == 9: # webcam index
            if _last_poll.get(i) is val:
                returns.append(gr.skip())
            else:
                returns.append(val)
                _last_poll[i] = val
        else:
            if _last_poll.get(i) == val:
                returns.append(gr.skip())
            else:
                returns.append(val)
                _last_poll[i] = val

    return tuple(returns)


COMPANY_CHOICES = [
    "DE Shaw", "Citadel", "Two Sigma", "Jane Street", "Jump Trading", 
    "Tower Research Capital", "Arcesium", "Bloomberg", "WorldQuant", "Trexquant",
    "Goldman Sachs", "Morgan Stanley", "JP Morgan Chase", "Barclays", "Visa", "Mastercard", "PayPal", "Stripe", "Square / Block", "Intuit",
    "Google", "Meta", "Apple", "Amazon", "Microsoft", "Netflix", "NVIDIA", "Adobe", "Salesforce", "Uber",
    "Oracle", "Cloudflare", "Cisco Systems", "Palo Alto Networks", "Datadog", "Snowflake", "MongoDB", "VMware", "Red Hat", "Akamai Technologies",
    "Flipkart", "Zomato", "Swiggy", "Razorpay", "Atlassian", "LinkedIn", "Airbnb", "Expedia", "McKinsey / Tech Consulting", "Boston Consulting Group (BCG)"
]

# ── Gradio UI ─────────────────────────────────────────────────────────
with gr.Blocks(title="INTERVION") as app:

    gr.Markdown("<h1 style='text-align: center;'>🎯 INTERVION — AI Interview System</h1>")
    gr.Markdown("<p style='text-align: center;'><i>Fully local AI mock interviewer powered by Whisper + Qwen 2.5</i></p>")

    # ── SCREEN 1: SETUP ───────────────────────────────────────────
    with gr.Column(visible=True) as setup_screen:
        gr.Markdown("### ⚙️ Interview Configuration")
        with gr.Row():
            with gr.Column():
                resume_file  = gr.File(label="Upload Resume (PDF)", type="filepath")
            with gr.Column():
                companies    = gr.Dropdown(
                                choices=COMPANY_CHOICES,
                                value=["DE Shaw", "Google"],
                                multiselect=True,
                                label="Target Companies (Select one or multiple)"
                               )
                roles        = gr.Textbox(label="Target Roles",      placeholder="ML Engineer, Backend Engineer")
                level        = gr.Dropdown(
                                choices=["intern", "entry", "mid", "senior"],
                                value="entry",
                                label="Experience Level"
                               )
                duration     = gr.Slider(10, 60, value=20, step=5, label="Duration (minutes)")
                focus_weak   = gr.Checkbox(value=True, label="🎯 Focus on previous weak areas (from past session reports)")

        with gr.Row():
            start_btn = gr.Button("🚀 Start Interview", variant="primary", size="lg")
        status_box = gr.Textbox(label="Status", interactive=False, value="Ready", visible=False)

    # ── SCREEN 2: LOADING ─────────────────────────────────────────
    with gr.Column(visible=False) as loading_screen:
        gr.Markdown("<br><br><br><h2 style='text-align: center;'>⏳ Loading AI Model & Preparing Interview...</h2>")
        gr.Markdown("<p style='text-align: center;'>Please wait ~10 seconds. The interview will start automatically.</p>")

    # ── SCREEN 3: INTERVIEW ───────────────────────────────────────
    with gr.Column(visible=False) as interview_screen:
        with gr.Row():
            # Left panel - Stats and controls
            with gr.Column(scale=1):
                state_box = gr.Textbox(
                              label="Current State",
                              value="Waiting...",
                              interactive=False,
                            )
                anxiety_slider    = gr.Slider(0, 100, value=0, label="😰 Anxiety",    interactive=False)
                confidence_slider = gr.Slider(0, 100, value=0, label="💪 Confidence", interactive=False)
                
                gr.Markdown("<br>")
                webcam_img = gr.Image(label="Your Camera", interactive=False, height=150)
                
                gr.Markdown("<br>")
                stop_btn  = gr.Button("⏹ End Interview", variant="stop")

            # Center - Avatar
            with gr.Column(scale=2, elem_id="avatar_col"):
                avatar_img  = gr.Image(
                                value=os.path.join(ASSETS, AVATAR_FRAMES["idle"][0]),
                                label="Interviewer",
                                interactive=False,
                                show_label=False
                              )

        # Bottom - collapsible transcript/history
        with gr.Accordion("📝 Live Transcript", open=False):
            transcript_box = gr.Textbox(
                               label="Transcript",
                               show_label=False,
                               lines=8,
                               interactive=False,
                             )
        with gr.Accordion("📋 Q&A History", open=False):
            history_md = gr.Markdown("No answers yet.")

    # ── SCREEN 4: REPORT CARD ─────────────────────────────────────
    with gr.Column(visible=False) as report_screen:
        gr.Markdown("<br><h2 style='text-align: center;'>🏆 Final Evaluation & Report Card</h2>")
        gr.Markdown("<p style='text-align: center;'><i>AI Hiring Committee Assessment Grounded in Two-Engine RAG & Vision Analytics</i></p>")
        
        with gr.Row():
            with gr.Column(scale=1):
                score_box   = gr.Textbox(label="Overall Score", value="0 / 100", interactive=False)
                verdict_box = gr.Textbox(label="Hiring Verdict", value="Pending...", interactive=False)
            with gr.Column(scale=2):
                summary_box = gr.Textbox(label="Executive Summary", lines=4, interactive=False)
        
        with gr.Row():
            with gr.Column():
                strengths_box  = gr.Textbox(label="💪 Demonstrated Strengths", lines=5, interactive=False)
            with gr.Column():
                weaknesses_box = gr.Textbox(label="⚠️ Areas for Improvement", lines=5, interactive=False)

        with gr.Row():
            with gr.Column(scale=1):
                topic_scores_box   = gr.Textbox(label="📊 Competency Topic Breakdown", lines=4, interactive=False)
            with gr.Column(scale=1):
                recommendation_box = gr.Textbox(label="🎯 Key Recommendation for Next Round", lines=4, interactive=False)
        
        with gr.Row():
            restart_btn = gr.Button("🔄 Start New Interview", variant="primary", size="lg")

    # ── Timer — polls shared state every second ────────────────────────
    timer = gr.Timer(value=0.1)  # 10 FPS for smooth avatar animation
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
        inputs  = [resume_file, companies, roles, level, duration, focus_weak],
        outputs = [status_box], # Hidden box just to consume string output
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

