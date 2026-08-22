const setupForm = document.getElementById("setup-form");
const setupScreen = document.getElementById("setup-screen");
const loadingScreen = document.getElementById("loading-screen");
const interviewScreen = document.getElementById("interview-screen");
const reportScreen = document.getElementById("report-screen");
const orb = document.getElementById("avatar-orb");
const transcriptBox = document.getElementById("transcript-box");
const stopBtn = document.getElementById("stop-btn");
const statusBadge = document.getElementById("status-badge");

let ws = null;
let mediaRecorder = null;
let audioContext = null;
let nextPlayTime = 0;

setupForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    
    // Switch to loading screen
    setupScreen.classList.add("hidden");
    loadingScreen.classList.remove("hidden");
    
    const formData = new FormData();
    const resumeFile = document.getElementById("resume").files[0];
    if (resumeFile) formData.append("resume", resumeFile);
    formData.append("companies", document.getElementById("companies").value);
    formData.append("roles", document.getElementById("roles").value);
    formData.append("level", document.getElementById("level").value);
    formData.append("llm_backend", document.getElementById("llm_backend").value);
    formData.append("llm_api_key", document.getElementById("api_key").value);
    
    try {
        const response = await fetch("/api/setup", {
            method: "POST",
            body: formData
        });
        const data = await response.json();
        
        if (data.status === "success") {
            startInterview(data.config);
        } else {
            alert("Setup failed: " + JSON.stringify(data));
            location.reload();
        }
    } catch (err) {
        alert("Error connecting to server: " + err);
        location.reload();
    }
});

async function startInterview(config) {
    loadingScreen.classList.add("hidden");
    interviewScreen.classList.remove("hidden");
    
    // Setup AudioContext for playback
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    
    // Connect WebSocket
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${protocol}//${window.location.host}/api/interview`);
    
    ws.onopen = async () => {
        statusBadge.textContent = "Connected";
        statusBadge.className = "px-3 py-1 rounded-full bg-green-600 text-white text-sm";
        // Send config
        ws.send(JSON.stringify(config));
        
        // Setup Mic
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
            mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
            
            mediaRecorder.ondataavailable = async (e) => {
                if (e.data.size > 0 && ws.readyState === WebSocket.OPEN) {
                    // Convert WebM to raw PCM 16kHz via AudioContext
                    const arrayBuffer = await e.data.arrayBuffer();
                    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                    // Extract channel 0
                    const pcmData = audioBuffer.getChannelData(0);
                    
                    // Downsample to 16kHz
                    const targetSampleRate = 16000;
                    const ratio = audioBuffer.sampleRate / targetSampleRate;
                    const newLength = Math.round(pcmData.length / ratio);
                    const result = new Float32Array(newLength);
                    
                    for (let i = 0; i < newLength; i++) {
                        result[i] = pcmData[Math.round(i * ratio)];
                    }
                    
                    // Send to WS
                    ws.send(result.buffer);
                }
            };
            
            // Collect chunks every 250ms
            mediaRecorder.start(250);
            
        } catch (err) {
            alert("Microphone access denied or error: " + err);
        }
    };
    
    ws.onmessage = async (event) => {
        if (event.data instanceof Blob) {
            // Received TTS Audio
            const arrayBuffer = await event.data.arrayBuffer();
            audioContext.decodeAudioData(arrayBuffer, (buffer) => {
                const source = audioContext.createBufferSource();
                source.buffer = buffer;
                source.connect(audioContext.destination);
                
                const currentTime = audioContext.currentTime;
                if (nextPlayTime < currentTime) {
                    nextPlayTime = currentTime;
                }
                
                source.start(nextPlayTime);
                nextPlayTime += buffer.duration;
            });
            return;
        }
        
        const msg = JSON.parse(event.data);
        if (msg.type === "state") {
            setOrbState(msg.state);
        } else if (msg.type === "transcript") {
            addTranscript(msg.text, msg.is_final);
        } else if (msg.type === "report") {
            showReport(msg.report);
        }
    };
    
    ws.onclose = () => {
        statusBadge.textContent = "Disconnected";
        statusBadge.className = "px-3 py-1 rounded-full bg-red-600 text-white text-sm";
        if (mediaRecorder) mediaRecorder.stop();
    };
}

function setOrbState(state) {
    const textEl = document.getElementById("ai-status-text");
    if (state === "loading" || state === "evaluating") {
        orb.className = "orb orb-thinking";
        textEl.textContent = state === "loading" ? "Thinking..." : "Evaluating...";
    } else if (state === "ai_speaking") {
        orb.className = "orb orb-talking";
        textEl.textContent = "AI is speaking";
    } else if (state === "listening") {
        orb.className = "orb orb-listening";
        textEl.textContent = "Listening...";
    } else if (state === "question_asked") {
        orb.className = "orb orb-idle";
        textEl.textContent = "Waiting for you to speak...";
    }
}

function addTranscript(text, isFinal) {
    if (!text.trim()) return;
    
    // Check if the last element is a partial transcript
    const lastEl = transcriptBox.lastElementChild;
    if (lastEl && lastEl.classList.contains("partial")) {
        transcriptBox.removeChild(lastEl);
    }
    
    const div = document.createElement("div");
    div.className = "p-3 rounded-lg " + (text.startsWith("??") ? "bg-blue-900/50 text-blue-100" : "bg-gray-800 text-gray-200");
    if (!isFinal) {
        div.classList.add("partial", "opacity-50");
    }
    div.textContent = text;
    
    transcriptBox.appendChild(div);
    transcriptBox.scrollTop = transcriptBox.scrollHeight;
}

function showReport(report) {
    interviewScreen.classList.add("hidden");
    reportScreen.classList.remove("hidden");
    
    document.getElementById("report-score").textContent = `${report.overall_score || 0}/100`;
    document.getElementById("report-verdict").textContent = report.verdict || "Complete";
    document.getElementById("report-summary").textContent = report.summary || "";
}

stopBtn.addEventListener("click", () => {
    if (ws) {
        ws.send(JSON.stringify({type: "stop"}));
    }
});

