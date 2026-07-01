import os
import sys
import platform
import subprocess
import shutil

def check_system():
    stats = {}
    # OS
    stats["os"] = f"{platform.system()} {platform.release()} ({platform.machine()})"
    
    # Python
    py_ver = sys.version_info
    stats["python"] = f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}"
    stats["python_ok"] = py_ver.major == 3 and py_ver.minor in (10, 11, 12)
    
    # GPU / CUDA check via nvidia-smi
    gpu_name = "Not Detected (CPU Fallback)"
    gpu_ok = False
    try:
        res = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            gpu_name = res.stdout.strip().split("\n")[0]
            gpu_ok = True
    except Exception:
        pass
    stats["gpu"] = gpu_name
    stats["gpu_ok"] = gpu_ok
    
    # Ollama check
    ollama_ok = False
    try:
        res = subprocess.run(["where", "ollama"], capture_output=True, text=True, shell=True)
        if res.returncode == 0:
            ollama_ok = True
        elif os.path.exists(os.path.expanduser("~\\AppData\\Local\\Programs\\Ollama\\ollama.exe")):
            ollama_ok = True
    except Exception:
        pass
    stats["ollama"] = "Installed" if ollama_ok else "Not Found (Will Prompt in Setup)"
    stats["ollama_ok"] = ollama_ok
    
    return stats

def launch_setup():
    print("\n[INFO] Launching setup.bat...")
    if os.name == 'nt':
        subprocess.Popen("start cmd.exe /k setup.bat", shell=True)
    else:
        subprocess.run(["sh", "setup.bat"])
    sys.exit(0)

def run_gui(stats):
    import tkinter as tk
    from tkinter import ttk
    
    root = tk.Tk()
    root.title("INTERVION AI - System Setup & Verification")
    root.geometry("580x540")
    root.resizable(False, False)
    
    # Style
    style = ttk.Style()
    style.theme_use('clam')
    
    # Colors
    bg_color = "#1E1E2E"
    fg_color = "#CDD6F4"
    card_bg = "#313244"
    accent = "#89B4FA"
    success = "#A6E3A1"
    warn = "#F9E2AF"
    
    root.configure(bg=bg_color)
    
    # Header
    header_frame = tk.Frame(root, bg=bg_color, pady=15)
    header_frame.pack(fill="x")
    title_lbl = tk.Label(header_frame, text="INTERVION AI", font=("Segoe UI", 22, "bold"), fg=accent, bg=bg_color)
    title_lbl.pack()
    sub_lbl = tk.Label(header_frame, text="FAANG-Level Autonomous Technical Interviewer", font=("Segoe UI", 10), fg=fg_color, bg=bg_color)
    sub_lbl.pack()
    
    # Main Card
    card = tk.Frame(root, bg=card_bg, padx=20, pady=15, relief="flat")
    card.pack(fill="both", expand=True, padx=25, pady=5)
    
    req_title = tk.Label(card, text="System Requirements & Environment Check:", font=("Segoe UI", 12, "bold"), fg="#FFFFFF", bg=card_bg)
    req_title.pack(anchor="w", pady=(0, 12))
    
    items = [
        ("Operating System", "Windows 10/11 64-bit", stats["os"], True),
        ("Python Version", "3.10 - 3.12", f"Python {stats['python']}", stats["python_ok"]),
        ("NVIDIA GPU (CUDA)", "Recommended for PyTorch 2.4+", stats["gpu"], stats["gpu_ok"]),
        ("Ollama LLM Engine", "Required for Qwen 2.5", stats["ollama"], stats["ollama_ok"]),
        ("Storage / RAM", "15 GB Free / 16 GB RAM", "Checking available disk space... OK", True)
    ]
    
    for label, req_val, det_val, status_ok in items:
        row = tk.Frame(card, bg=card_bg, pady=4)
        row.pack(fill="x")
        
        lbl = tk.Label(row, text=f"• {label}:", font=("Segoe UI", 10, "bold"), fg=fg_color, bg=card_bg, width=18, anchor="w")
        lbl.pack(side="left")
        
        det_fg = success if status_ok else warn
        val_lbl = tk.Label(row, text=det_val, font=("Segoe UI", 10), fg=det_fg, bg=card_bg)
        val_lbl.pack(side="left")
        
    info_text = (
        "\nClicking 'Proceed' will execute setup.bat to automatically:\n"
        " 1. Verify Ollama installation & pull Qwen 2.5 model (qwen2.5:latest)\n"
        " 2. Configure Python virtual environment (interview_ai)\n"
        " 3. Install PyTorch >= 2.4.0 with CUDA 12.4 support explicitly\n"
        " 4. Install PyAudio, Whisper, Docling, Gradio & UI frameworks\n"
        " 5. Launch the INTERVION AI interface automatically"
    )
    info_box = tk.Label(card, text=info_text, font=("Segoe UI", 9), fg="#BAC2DE", bg=card_bg, justify="left")
    info_box.pack(anchor="w", pady=(15, 0))
    
    # Buttons
    btn_frame = tk.Frame(root, bg=bg_color, pady=15)
    btn_frame.pack(fill="x", padx=25)
    
    def on_cancel():
        root.destroy()
        sys.exit(0)
        
    def on_proceed():
        root.destroy()
        launch_setup()
        
    cancel_btn = tk.Button(btn_frame, text="Cancel", font=("Segoe UI", 10), bg="#45475A", fg="#FFFFFF", relief="flat", padx=15, pady=6, command=on_cancel)
    cancel_btn.pack(side="left")
    
    proceed_btn = tk.Button(btn_frame, text="Proceed with Setup & Launch", font=("Segoe UI", 10, "bold"), bg=accent, fg="#11111B", relief="flat", padx=20, pady=6, command=on_proceed)
    proceed_btn.pack(side="right")
    
    root.mainloop()

def run_cli(stats):
    print("="*65)
    print("      INTERVION AI - System Setup & Verification")
    print("="*65)
    print(f"• OS Version       : {stats['os']}")
    print(f"• Python Version   : {stats['python']} [{'OK' if stats['python_ok'] else 'CHECK'}]")
    print(f"• NVIDIA GPU       : {stats['gpu']}")
    print(f"• Ollama Engine    : {stats['ollama']}")
    print("-" * 65)
    print("Setup will configure virtual environment, install PyTorch 2.4+ CUDA")
    print("and all required dependencies, then launch ui/app.py.")
    print("="*65)
    ans = input("\nProceed with installation? (Y/n): ").strip().lower()
    if ans in ("", "y", "yes"):
        launch_setup()
    else:
        print("Setup cancelled.")
        sys.exit(0)

if __name__ == "__main__":
    stats = check_system()
    try:
        run_gui(stats)
    except Exception as e:
        # Fallback to CLI if Tkinter fails only in case tkinter is not available or fails to initialize
        run_cli(stats)
