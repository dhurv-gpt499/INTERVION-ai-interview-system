import gradio as gr
import threading
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_processor.pipeline import run_pipeline
from resume_parser.resume_parser import parse_resume

# ── Avatar image paths ─────────────────────────────────────────────────
ASSETS = os.path.join(os.path.dirname(__file__), "assets")
AVATAR = {
    "idle"      : os.path.join(ASSETS, "avatar_idle.gif"),
    "talking"   : os.path.join(ASSETS, "avatar_talking.gif"),
    "listening" : os.path.join(ASSETS, "avatar_listening.gif"),
}

# ── Shared state (pipeline writes, UI reads) ──────────────────────────
shared = {
    "screen"      : "setup", # "setup", "loading", "interview"
    "transcript"  : "",
    "state"       : "Waiting to start...",
    "avatar"      : "idle",
    "anxiety"     : 0.0,
    "confidence"  : 100.0,
    "webcam"      : None,
    "qa_history"  : [],
    "running"     : False,
}

pipeline_thread = None

# ── Pipeline callbacks (called from pipeline thread) ──────────────────
def on_state_change(state_name: str):
    shared["state"] = state_name
    if state_name in ("ai_speaking",):
        shared["avatar"] = "talking"
    elif state_name in ("listening", "candidate_paused"):
        shared["avatar"] = "listening"
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
    shared["webcam"] = frame


# ── Start interview ────────────────────────────────────────────────────
def start_interview(resume_file, companies_str, roles_str, level, duration):
    global pipeline_thread

    if shared["running"]:
        return "Already running!"

    if resume_file is None:
        return "Please upload a resume first."

    file_path = resume_file.name if hasattr(resume_file, "name") else str(resume_file)

    companies = [c.strip() for c in companies_str.split(",") if c.strip()]
    roles     = [r.strip() for r in roles_str.split(",") if r.strip()]

    # reset shared state
    shared["transcript"] = ""
    shared["state"]      = "Starting..."
    shared["avatar"]     = "idle"
    shared["qa_history"] = []
    shared["running"]    = True
    shared["screen"]     = "loading"
    shared["state"]      = "Parsing Resume (takes ~15s)..."

    # run pipeline in background thread
    pipeline_thread = threading.Thread(
        target      = _run_pipeline_thread,
        args        = (file_path, companies, roles, level, int(duration)),
        daemon      = True,
    )
    pipeline_thread.start()
    return "Starting..."


def _run_pipeline_thread(resume_file_path, companies, roles, level, duration):
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
            on_state_change     = on_state_change,
            on_transcript       = on_transcript_update,
            on_qa_complete      = on_qa_complete,
            on_vision_scores    = on_vision_scores,
            on_vision_frame     = on_vision_frame,
            is_running          = lambda: shared["running"],
        )
    finally:
        shared["running"] = False
        shared["state"]   = "Interview complete."
        shared["avatar"]  = "idle"
        shared["screen"]  = "setup"


def stop_interview():
    shared["running"] = False
    shared["state"]   = "Stopped."
    shared["avatar"]  = "idle"
    shared["screen"]  = "setup"
    return "Stopped."


# ── Polling — updates UI every second ─────────────────────────────────
def poll():
    # Check if we should move from loading to interview screen
    if shared["screen"] == "loading" and shared["running"]:
        if shared["state"] not in ("Starting...", "Loading AI Model (takes ~10s)...", "Ready"):
            shared["screen"] = "interview"

    # Screen visibility
    show_setup = gr.update(visible=(shared["screen"] == "setup"))
    show_loading = gr.update(visible=(shared["screen"] == "loading"))
    show_interview = gr.update(visible=(shared["screen"] == "interview"))

    avatar_img = AVATAR.get(shared["avatar"], AVATAR["idle"])
    history_md = "\n\n".join(
        f"**Q{i+1}:** {qa['q']}\n\n**A:** {qa['a']}"
        for i, qa in enumerate(shared["qa_history"])
    ) or "No answers yet."

    return (
        show_setup,
        show_loading,
        show_interview,
        avatar_img,
        shared["state"],
        shared["transcript"].strip(),
        shared["anxiety"],
        shared["confidence"],
        shared["webcam"],
        history_md,
    )


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
                companies    = gr.Textbox(label="Target Companies",  placeholder="Google, Microsoft, Amazon")
                roles        = gr.Textbox(label="Target Roles",      placeholder="ML Engineer, Backend Engineer")
                level        = gr.Dropdown(
                                choices=["intern", "entry", "mid", "senior"],
                                value="entry",
                                label="Experience Level"
                               )
                duration     = gr.Slider(10, 60, value=20, step=5, label="Duration (minutes)")

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
                                value=AVATAR["idle"],
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

    # ── Timer — polls shared state every second ────────────────────────
    timer = gr.Timer(value=1)
    timer.tick(
        fn      = poll,
        outputs = [setup_screen, loading_screen, interview_screen,
                   avatar_img, state_box, transcript_box,
                   anxiety_slider, confidence_slider, webcam_img, history_md],
    )

    # ── Button events ──────────────────────────────────────────────────
    start_btn.click(
        fn      = start_interview,
        inputs  = [resume_file, companies, roles, level, duration],
        outputs = [status_box], # Hidden box just to consume string output
    )

    stop_btn.click(
        fn      = stop_interview,
        outputs = [status_box],
    )


if __name__ == "__main__":
    app.launch(share=False)
