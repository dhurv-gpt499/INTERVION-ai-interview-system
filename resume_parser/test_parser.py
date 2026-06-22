from resume_parser import extract_text, extract_personal_info, extract_education
if __name__ == "__main__":
    # Change "your_resume.pdf" to your actual resume filename
    text = extract_text("resume_parser/test_resume.pdf")
    print("=== RAW TEXT (first 500 chars) ===")
    print("=== RAW TEXT ===")
    print(text[:1000])
    
    print("\n=== PERSONAL INFO ===")
    personal = extract_personal_info(text)
    print(personal)
    
    print("\n=== EDUCATION ===")
    education = extract_education(text)
    print(education)