import sqlite3
import json
from pathlib import Path
from datetime import datetime

# Database file will be created in the database/ folder
DB_PATH = Path(__file__).parent / "intervion.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets us access columns by name like dict
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS resumes (
        email       TEXT PRIMARY KEY,
        name        TEXT NOT NULL,
        phone       TEXT,
        linkedin    TEXT,
        github      TEXT,
        uploaded_at TEXT
    );
    
    CREATE TABLE IF NOT EXISTS resume_raw (
        email      TEXT PRIMARY KEY,
        raw_text   TEXT,
        raw_pdf    BLOB,
        uploaded_at TEXT,
        FOREIGN KEY (email) REFERENCES resumes(email)
    );
    
    CREATE TABLE IF NOT EXISTS resume_parsed (
        email           TEXT PRIMARY KEY,
        education       TEXT,
        skills          TEXT,
        experience      TEXT,
        projects        TEXT,
        achievements    TEXT,
        competitive     TEXT,
        full_parsed     TEXT,
        parsed_at       TEXT,
        FOREIGN KEY (email) REFERENCES resumes(email)
    );
    
    CREATE TABLE IF NOT EXISTS sessions (
        session_id    TEXT PRIMARY KEY,
        resume_email  TEXT NOT NULL,
        started_at    TEXT,
        completed_at  TEXT,
        overall_score REAL,
        session_data  TEXT,
        FOREIGN KEY (resume_email) REFERENCES resumes(email)
    );
    """)
    
    conn.commit()
    conn.close()
    print("Database initialized successfully!")
    

def save_name(name: str, email: str) -> bool:
    """Save name as soon as user types it in Streamlit — before parsing."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO resumes (email, name, uploaded_at)
            VALUES (?, ?, ?)
        """, (email, name, datetime.now().isoformat()))
        
        conn.commit()
        print(f"Name '{name}' saved for {email}")
        return True
    
    except Exception as e:
        print(f"Error saving name: {e}")
        return False
    
    finally:
        conn.close()

def update_parsed_data(email: str, parsed_data: dict, pdf_path: str) -> bool:
    """Update resume row with parsed JSON after extraction is complete."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            UPDATE resumes
            SET phone       = ?,
                linkedin    = ?,
                github      = ?,
                pdf_path    = ?,
                parsed_json = ?
            WHERE email = ?
        """, (
            parsed_data["personal_info"]["phone"],
            parsed_data["personal_info"]["linkedin"],
            parsed_data["personal_info"]["github"],
            pdf_path,
            json.dumps(parsed_data),
            email
        ))
        
        conn.commit()
        print(f"Parsed data updated for {email}")
        return True
    
    except Exception as e:
        print(f"Error updating parsed data: {e}")
        return False
    
    finally:
        conn.close()
        

def save_raw_resume(email: str, raw_text: str, pdf_bytes: bytes) -> bool:
    """Store full resume text and PDF blob in database."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO resume_raw (email, raw_text, raw_pdf, uploaded_at)
            VALUES (?, ?, ?, ?)
        """, (
            email,
            raw_text,
            pdf_bytes,              # actual PDF bytes stored as BLOB
            datetime.now().isoformat()
        ))
        
        conn.commit()
        print(f"Raw resume stored for {email}")
        return True
    
    except Exception as e:
        print(f"Error storing raw resume: {e}")
        return False
    
    finally:
        conn.close()

def save_parsed_sections(email: str, parsed_data: dict) -> bool:
    """Store each parsed section separately for easy retrieval during interview."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO resume_parsed 
            (email, education, skills, experience, projects, achievements, competitive, full_parsed, parsed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            email,
            json.dumps(parsed_data.get("education", [])),
            json.dumps(parsed_data.get("skills", {})),
            json.dumps(parsed_data.get("experience", [])),
            json.dumps(parsed_data.get("projects", [])),
            json.dumps(parsed_data.get("achievements", [])),
            json.dumps(parsed_data.get("competitive_programming", {})),
            json.dumps(parsed_data),        # full data as backup
            datetime.now().isoformat()
        ))
        
        conn.commit()
        print(f"Parsed sections saved for {email}")
        return True
    
    except Exception as e:
        print(f"Error saving parsed sections: {e}")
        return False
    
    finally:
        conn.close()

def get_parsed_resume(email: str) -> dict:
    """Fetch parsed resume sections for use during interview."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM resume_parsed WHERE email = ?", (email,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        data = dict(row)
        # Convert each JSON string back to Python object
        for field in ["education", "skills", "experience", "projects", "achievements", "competitive", "full_parsed"]:
            data[field] = json.loads(data[field]) if data[field] else {}
        return data
    return None


if __name__ == "__main__":
    init_db()
    