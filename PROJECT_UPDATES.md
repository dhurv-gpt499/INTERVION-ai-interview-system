# INTERVION - Project Update & Fixes Report

This document outlines all the architectural changes, latency optimizations, and bug fixes applied to the INTERVION AI Interview System.

## 🛠 What Was Fixed & Built

### 1. Massive Latency Reduction (VAD)
- **Problem**: The Voice Activity Detection (VAD) system was waiting 5 full seconds after the candidate stopped speaking before transcribing, causing a huge delay in the AI's response.
- **Fix**: Reduced the `silence_frames_threshold` in `vad.py` from 156 frames (5s) to 60 frames (2s). The pipeline is now highly responsive and snappy.

### 2. Audio & Animation Synchronization ("Ghost Talking")
- **Problem**: The AI avatar would switch to the "talking" animation, but no audio would play. This happened because the Text-To-Speech (TTS) engine was trying to generate audio for standalone punctuation marks (like `"."` or `"...""`), which Edge-TTS converts into blank audio files.
- **Fix**: Added a strict alphanumeric Regex filter in `tts_engine.py`. The avatar now *only* animates when there is actual phonetic speech being played.

### 3. Removal of Fake "Buzz Words"
- **Problem**: The system was hardcoded to inject filler words like "Interesting..." before every response, even if the candidate was silent or struggling.
- **Fix**: Completely stripped the `add_filler` injection logic from `interviewer.py`. The AI now speaks naturally and only when it has a targeted question or hint to provide.

### 4. Crisp Visual Assets (No Motion Blur)
- **Problem**: The avatar animations looked blurry and unprofessional because of a crossfading script designed for photos, not 2D vector art. The `.gif` format was also causing Gradio to crash due to transparency issues.
- **Fix**: Rewrote `generate_gifs.py` to strip the image blending logic and output crisp, snappy frame-by-frame animations in the highly efficient Animated `.webp` format.

### 5. Authentic Company Personas
- **Problem**: The AI behaved like a polite, generic chatbot rather than a high-stakes technical interviewer at a specific firm.
- **Fix**: Rewrote `build_system_prompt.py` to detect the target company and inject a strict behavioral persona. For example, if interviewing for a Quant firm, the AI becomes hyper-analytical and interrupts with edge cases. If FAANG, it demands Big-O complexity analysis.

### 6. The Question Bank Datastore (RAG Injection)
- **Problem**: The LLM was inventing its own questions on the fly, which resulted in generic textbook trivia (e.g., "What is a Hash Map?") instead of real interview problems.
- **Fix**: Built a massive `database/question_bank.json` populated with actual LeetCode-style DSA algorithms, System Design scaling challenges, and classic Quant Brainteasers.
- **RAG Integration**: Modified the prompt builder so that it randomly selects a real problem based on the candidate's agenda, and dynamically injects the exact problem statement *and the optimal grading rubric* directly into the LLM's brain.

---

## 💻 Tech Stack Used

- **Backend Logic**: Python 3
- **User Interface**: Gradio (Web App)
- **Text-to-Speech**: `edge-tts` (Microsoft Azure Cognitive Services)
- **Audio Playback**: `pygame`
- **Transcription**: OpenAI Whisper (via `transcriber.py`)
- **Large Language Model**: Ollama (`qwen2.5:latest`)
- **RAG / NLP**: `scikit-learn` (TF-IDF Vectorization & Cosine Similarity)
- **Image Processing**: `Pillow` (PIL) for WebP animation synthesis
- **Datastore**: SQLite (`intervion.db`) & JSON (`question_bank.json`)

---

*All systems are fully updated, wired together, and ready for production testing via `python ui/app.py`.*
