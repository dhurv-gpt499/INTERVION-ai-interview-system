from llm_interviewer.rag_engine import ResumeRAG

def test():
    resume = {
        "experience": [
            {"company": "Google", "role": "SWE", "description": "Built scalable APIs."}
        ],
        "projects": [
            {"name": "Deepfake Detector", "description": "Used XceptionNet to detect deepfakes."}
        ]
    }
    
    rag = ResumeRAG()
    rag.build_index(resume)
    
    ctx = rag.get_relevant_context("Tell me about your machine learning experience.")
    print("Retrieved Context:")
    print(ctx)

if __name__ == "__main__":
    test()
