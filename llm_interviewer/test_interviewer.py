import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_interviewer.interviewer import QwenInterviewer

# --- Mock resume (no need to parse a PDF for testing) ---
mock_resume = {
    "education"   : "B.Tech Computer Science, 2025",
    "skills"      : "Python, PyTorch, Machine Learning, FastAPI, SQL",
    "experience"  : "Intern at XYZ Corp - built a deepfake detection model",
    "projects"    : "Deepfake detection using XceptionNet, Resume parser using Qwen",
    "achievements": "Top 5% on Codeforces, 3 star CodeChef",
    "competitive" : "Codeforces rating 1450"
}

def stream_to_console(generator):
    """Print tokens as they arrive, return full response."""
    full = ""
    for token in generator:
        print(token, end="", flush=True)
        full += token
    print()  # newline after response
    return full

# --- Run test ---
interviewer = QwenInterviewer()

print("=" * 60)
print("Starting interview...")
print("=" * 60)
print("\n[INTERVIEWER]: ", end="")

# Start — get opening question
stream_to_console(
    interviewer.start(
        resume_parsed       = mock_resume,
        preferred_companies = ["Google", "Microsoft"],
        preferred_roles     = ["ML Engineer", "Backend Engineer"],
        target_level        = "entry",
        domain              = "software engineering",
        duration_minutes    = 20,
    )
)

# Simulate 3 turns of conversation
test_answers = [
    "I have been working with Python for 3 years. I used it mainly for machine learning projects with PyTorch.",
    "In my deepfake detection project I used XceptionNet as the backbone. It gave me 94 percent accuracy on the dataset.",
    "I am comfortable with REST APIs. I have built a few with FastAPI and Flask.",
]

for i, answer in enumerate(test_answers, 1):
    print(f"\n[CANDIDATE]: {answer}")
    print(f"\n[INTERVIEWER]: ", end="")
    stream_to_console(interviewer.receive_answer(answer))

print("\n" + "=" * 60)
print(f"Turns completed: {interviewer.turn_count}")
print(f"Elapsed: {interviewer.elapsed_minutes():.1f} min")
print(f"Interview active: {interviewer.is_active}")
print("\nConversation History:")
for qa in interviewer.get_history():
    print(f"\nQ: {qa['question'][:80]}...")
    print(f"A: {qa['answer'][:80]}...")