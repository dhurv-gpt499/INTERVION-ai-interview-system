from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import json
import asyncio
import os
import sys

# Ensure parent directory is in path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resume_parser.resume_parser import parse_resume

app = FastAPI(title="INTERVION 2.0 API")

# Mount static files for the frontend
client_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "client")
app.mount("/static", StaticFiles(directory=os.path.join(client_dir, "static")), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open(os.path.join(client_dir, "index.html"), "r") as f:
        return f.read()

@app.post("/api/setup")
async def setup_interview(
    resume: UploadFile = File(None),
    companies: str = Form(""),
    roles: str = Form(""),
    level: str = Form("Mid-Level"),
    duration: int = Form(20),
    focus_weak: bool = Form(True),
    llm_backend: str = Form("Cloud API (Groq)"),
    llm_api_key: str = Form("")
):
    """
    Parses the resume and returns the initial interview configuration.
    """
    resume_parsed = {}
    if resume:
        # Save temp file
        temp_path = f"temp_{resume.filename}"
        with open(temp_path, "wb") as f:
            f.write(await resume.read())
        try:
            resume_parsed, _ = parse_resume(temp_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse resume: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    company_list = [c.strip() for c in companies.split(",") if c.strip()]
    role_list = [r.strip() for r in roles.split(",") if r.strip()]

    return {
        "status": "success",
        "config": {
            "resume_parsed": resume_parsed,
            "companies": company_list,
            "roles": role_list,
            "level": level,
            "duration": duration,
            "focus_weak": focus_weak,
            "llm_backend": llm_backend,
            "llm_api_key": llm_api_key
        }
    }

@app.websocket("/api/interview")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connected")
    
    # We will import the async pipeline handler here
    from v2.server.pipeline_async import handle_interview_session
    
    try:
        # Wait for the initial config message from the client
        config_msg = await websocket.receive_text()
        config = json.loads(config_msg)
        
        await handle_interview_session(websocket, config)
        
    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass

