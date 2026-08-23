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
let isFinished = false;
let nextPlayTime = 0;
let activeSources = [];

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
        statusBadge.textContent = "Analyzing Resume & Formulating Strategy...";
        statusBadge.className = "px-3 py-1 rounded-full bg-purple-600 text-white text-sm font-semibold animate-pulse";
        // Send config
        ws.send(JSON.stringify(config));
        
        // Setup Mic
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: { echoCancellation: true, noiseSuppression: true }, 
                video: false 
            });
            
            const source = audioContext.createMediaStreamSource(stream);
            const processor = audioContext.createScriptProcessor(4096, 1, 1);
            
            // Zero gain node to prevent echo loop while keeping the graph active
            const zeroGain = audioContext.createGain();
            zeroGain.gain.value = 0;
            
            source.connect(processor);
            processor.connect(zeroGain);
            zeroGain.connect(audioContext.destination);
            
            processor.onaudioprocess = (e) => {
                if (ws.readyState === WebSocket.OPEN) {
                    const inputData = e.inputBuffer.getChannelData(0);
                    
                    // Downsample to 16kHz
                    const targetSampleRate = 16000;
                    const ratio = audioContext.sampleRate / targetSampleRate;
                    
                    let result;
                    if (ratio > 1.01 || ratio < 0.99) {
                        const newLength = Math.round(inputData.length / ratio);
                        result = new Float32Array(newLength);
                        for (let i = 0; i < newLength; i++) {
                            result[i] = inputData[Math.round(i * ratio)];
                        }
                    } else {
                        result = new Float32Array(inputData);
                    }
                    
                    ws.send(result.buffer);
                }
            };
            
            // Polyfill for mediaRecorder.stop()
            mediaRecorder = {
                stop: () => {
                    processor.disconnect();
                    zeroGain.disconnect();
                    source.disconnect();
                    stream.getTracks().forEach(t => t.stop());
                }
            };
            
        } catch (err) {
            alert("Microphone access denied or error: " + err);
        }
    };
    
    let expectingPlaybackComplete = false;

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
                    nextPlayTime = currentTime + 0.1;
                }
                
                source.start(nextPlayTime);
                nextPlayTime += buffer.duration;
                
                activeSources.push(source);
                source.onended = () => {
                    activeSources = activeSources.filter(s => s !== source);
                    if (activeSources.length === 0 && expectingPlaybackComplete) {
                        expectingPlaybackComplete = false;
                        ws.send(JSON.stringify({type: "playback_complete"}));
                    }
                };
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
        } else if (msg.type === "interrupt") {
            expectingPlaybackComplete = false;
            activeSources.forEach(s => {
                try { s.stop(); } catch (e) {}
            });
            activeSources = [];
            if (audioContext) {
                nextPlayTime = audioContext.currentTime;
            } else {
                nextPlayTime = 0;
            }
        } else if (msg.type === "vision") {
            const badge = document.getElementById("vision-badge");
            if (badge) {
                badge.classList.remove("hidden");
                badge.textContent = `Anxiety: ${Math.round(msg.anxiety)}% | Conf: ${Math.round(msg.confidence)}%`;
                
                // Change color based on anxiety
                if (msg.anxiety > 60) {
                    badge.className = "px-3 py-1 rounded-full bg-red-600 text-white text-sm font-medium transition-all duration-300";
                } else if (msg.anxiety > 30) {
                    badge.className = "px-3 py-1 rounded-full bg-yellow-600 text-white text-sm font-medium transition-all duration-300";
                } else {
                    badge.className = "px-3 py-1 rounded-full bg-indigo-600 text-white text-sm font-medium transition-all duration-300";
                }
            }
        } else if (msg.type === "tts_complete") {
            if (activeSources.length === 0) {
                ws.send(JSON.stringify({type: "playback_complete"}));
            } else {
                expectingPlaybackComplete = true;
            }
        }
    };
    ws.onclose = () => {
        statusBadge.textContent = "Disconnected";
        statusBadge.className = "px-3 py-1 rounded-full bg-gray-600 text-white text-sm";
        if (mediaRecorder) mediaRecorder.stop();
        
        // Show disconnected banner only if not gracefully finished
        if (!isFinished) {
            const banner = document.createElement("div");
            banner.className = "fixed top-0 left-0 w-full bg-red-600 text-white text-center py-2 z-50 font-semibold shadow-lg";
            banner.innerHTML = "Connection lost! The server may have restarted. <a href='/' class='underline hover:text-gray-200'>Click here to refresh</a>.";
            document.body.appendChild(banner);
        }
    };
}

function setOrbState(state) {
    const statusBadge = document.getElementById("status-badge");
    if (state === "loading") {
        orb.className = "orb orb-thinking";
        if (statusBadge) {
            statusBadge.textContent = "Analyzing Resume & Formulating Strategy...";
            statusBadge.className = "px-3 py-1 rounded-full bg-purple-600 text-white text-sm font-semibold animate-pulse";
        }
    } else if (state === "evaluating") {
        orb.className = "orb orb-thinking";
        if (statusBadge) {
            statusBadge.textContent = "Evaluating Report...";
            statusBadge.className = "px-3 py-1 rounded-full bg-yellow-600 text-white text-sm";
        }
    } else if (state === "ai_speaking") {
        orb.className = "orb orb-talking";
        if (statusBadge) {
            statusBadge.textContent = "AI Speaking";
            statusBadge.className = "px-3 py-1 rounded-full bg-blue-600 text-white text-sm";
        }
    } else if (state === "question_asked" || state === "listening") {
        orb.className = "orb orb-listening";
        if (statusBadge) {
            statusBadge.textContent = "Listening...";
            statusBadge.className = "px-3 py-1 rounded-full bg-green-600 text-white text-sm animate-pulse";
        }
    }
}

function addTranscript(text, isFinal) {
    if (!text.trim()) return;
    
    const isAI = text.startsWith("AI:");
    const cleanText = text.replace(/[*_`~#\[\]]/g, "").replace(/AI:/g, "").replace(/You:/g, "").trim();
    const targetBoxId = isAI ? "transcript-interviewer" : "transcript-you";
    const targetBox = document.getElementById(targetBoxId);
    
    // Check if the last element is a partial transcript
    const lastEl = targetBox.lastElementChild;
    if (lastEl && lastEl.classList.contains("partial")) {
        targetBox.removeChild(lastEl);
    }
    
    const div = document.createElement("div");
    div.className = "p-3 rounded-lg " + (isAI ? "bg-blue-900/50 text-blue-100" : "bg-gray-800 text-gray-200");
    if (!isFinal) {
        div.classList.add("partial", "opacity-50");
    }
    div.textContent = cleanText;
    
    targetBox.appendChild(div);
    targetBox.scrollTop = targetBox.scrollHeight;
}

// Tab Switching Logic
document.getElementById("tab-interviewer").addEventListener("click", (e) => {
    e.target.classList.add("text-blue-400", "border-b-2", "border-blue-400");
    e.target.classList.remove("text-gray-500");
    
    const youBtn = document.getElementById("tab-you");
    youBtn.classList.remove("text-blue-400", "border-b-2", "border-blue-400");
    youBtn.classList.add("text-gray-500");
    
    document.getElementById("transcript-interviewer").classList.remove("hidden");
    document.getElementById("transcript-you").classList.add("hidden");
});

document.getElementById("tab-you").addEventListener("click", (e) => {
    e.target.classList.add("text-blue-400", "border-b-2", "border-blue-400");
    e.target.classList.remove("text-gray-500");
    
    const intBtn = document.getElementById("tab-interviewer");
    intBtn.classList.remove("text-blue-400", "border-b-2", "border-blue-400");
    intBtn.classList.add("text-gray-500");
    
    document.getElementById("transcript-you").classList.remove("hidden");
    document.getElementById("transcript-interviewer").classList.add("hidden");
});

function showReport(report) {
    isFinished = true;
    document.getElementById("interview-screen").classList.add("hidden");
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

