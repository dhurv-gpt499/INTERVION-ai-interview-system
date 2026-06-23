from resume_parser import parse_resume
import json

result, raw_text = parse_resume("resume_parser/test_resume.pdf")
print(json.dumps(result, indent=2))