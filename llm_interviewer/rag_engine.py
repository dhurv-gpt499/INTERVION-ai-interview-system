import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ResumeRAG:
    def __init__(self):
        # Engine A: Resume TF-IDF Indexer
        self.resume_vectorizer = TfidfVectorizer(stop_words='english')
        self.resume_chunks = []
        self.resume_vectors = None

        # Engine B: Competency Rubrics TF-IDF Indexer
        self.rubric_vectorizer = TfidfVectorizer(stop_words='english')
        self.rubric_keys = []
        self.rubric_docs = []
        self.rubric_vectors = None
        self.rubrics_data = {}
        self.company_profiles = {}

        # Load Knowledge Graph
        self._load_knowledge_graph()

    def _load_knowledge_graph(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        rubrics_path = os.path.join(cur_dir, "rubrics_bank.json")
        companies_path = os.path.join(cur_dir, "company_profiles.json")

        if os.path.exists(rubrics_path):
            try:
                with open(rubrics_path, "r", encoding="utf-8") as f:
                    self.rubrics_data = json.load(f)
                
                self.rubric_keys = []
                self.rubric_docs = []
                for k, v in self.rubrics_data.items():
                    # Build rich text document for TF-IDF keyword matching
                    keywords_str = " ".join(v.get("keywords", []))
                    domain_str = v.get("domain", "")
                    q_str = v.get("core_question", "")
                    doc = f"{k} {domain_str} {keywords_str} {q_str}"
                    self.rubric_keys.append(k)
                    self.rubric_docs.append(doc)

                if self.rubric_docs:
                    self.rubric_vectors = self.rubric_vectorizer.fit_transform(self.rubric_docs)
                    print(f"[RAG] Built Rubric Index with {len(self.rubric_docs)} FAANG/Quant competency nodes.")
            except Exception as e:
                print(f"[RAG] Error loading rubrics_bank.json: {e}")

        if os.path.exists(companies_path):
            try:
                with open(companies_path, "r", encoding="utf-8") as f:
                    self.company_profiles = json.load(f)
                print(f"[RAG] Loaded {len(self.company_profiles)} company profiles.")
            except Exception as e:
                print(f"[RAG] Error loading company_profiles.json: {e}")

    def build_index(self, resume_parsed: dict):
        """
        Takes the enriched parsed resume JSON and builds Engine A (Resume TF-IDF index).
        """
        self.resume_chunks = []
        
        # 1. Enriched Candidate Intelligence Fields
        if resume_parsed.get("candidate_archetype"):
            self.resume_chunks.append(f"Candidate Archetype / Domain: {resume_parsed.get('candidate_archetype')}")
            
        if resume_parsed.get("flagship_project_tech"):
            tech_list = ", ".join(resume_parsed.get("flagship_project_tech", []))
            if tech_list:
                self.resume_chunks.append(f"Flagship Project Tech Stack: {tech_list}")
                
        if resume_parsed.get("probe_targets"):
            for probe in resume_parsed.get("probe_targets", []):
                self.resume_chunks.append(f"Grilling Target / Resume Claim to Probe: {probe}")

        # 2. Standard Resume Sections
        if "experience" in resume_parsed:
            for exp in resume_parsed.get("experience", []):
                desc = exp.get('description', '')
                if isinstance(desc, list):
                    desc = " ".join(desc)
                self.resume_chunks.append(f"Experience at {exp.get('company', '')} as {exp.get('role', '')}: {desc}")
                
        if "projects" in resume_parsed:
            for proj in resume_parsed.get("projects", []):
                desc = proj.get('description', '')
                if isinstance(desc, list):
                    desc = " ".join(desc)
                tech = ", ".join(proj.get('tech_stack', []))
                self.resume_chunks.append(f"Project {proj.get('name', '')} ({tech}): {desc}")
                
        if "skills" in resume_parsed:
            skills_dict = resume_parsed.get("skills", {})
            all_skills = []
            for k, v in skills_dict.items():
                if isinstance(v, list):
                    all_skills.extend(v)
            if all_skills:
                self.resume_chunks.append(f"Technical Skills: {', '.join(all_skills)}")
                
        if "education" in resume_parsed:
            for edu in resume_parsed.get("education", []):
                self.resume_chunks.append(f"Education: {edu.get('degree', '')} in {edu.get('branch', '')} at {edu.get('institution', '')} (CGPA: {edu.get('cgpa', '')})")

        if "competitive_programming" in resume_parsed:
            cp = resume_parsed.get("competitive_programming", {})
            cp_str = ", ".join([f"{k}: {v}" for k, v in cp.items() if v])
            if cp_str:
                self.resume_chunks.append(f"Competitive Programming Profiles: {cp_str}")

        if not self.resume_chunks:
            self.resume_chunks.append("Resume data is minimal.")

        # Build TF-IDF Vectors for Resume
        self.resume_vectors = self.resume_vectorizer.fit_transform(self.resume_chunks)
        print(f"[RAG] Built Resume Index with {len(self.resume_chunks)} enriched chunks.")

    def get_relevant_context(self, query: str, top_k_resume: int = 2, target_company: str = "") -> str:
        """
        Queries both Engine A (Resume) and Engine B (Rubrics Bank) to produce a unified FAANG-level context block.
        """
        output_blocks = []

        # -- Query Engine A: Resume Chunks --
        if self.resume_vectors is not None and self.resume_chunks:
            try:
                q_vec = self.resume_vectorizer.transform([query])
                sims = cosine_similarity(q_vec, self.resume_vectors).flatten()
                if not np.all(sims == 0):
                    top_indices = sims.argsort()[-top_k_resume:][::-1]
                    res_texts = [self.resume_chunks[i] for i in top_indices if sims[i] > 0.05]
                    if res_texts:
                        output_blocks.append("[CANDIDATE RESUME CONTEXT]:\n" + "\n".join([f"- {t}" for t in res_texts]))
            except Exception as e:
                print(f"[RAG] Resume query error: {e}")

        # -- Query Engine B: Competency Rubrics Bank --
        if self.rubric_vectors is not None and self.rubrics_data:
            try:
                # If target company exists, boost its primary_focus topics
                boosted_sims = None
                q_vec = self.rubric_vectorizer.transform([query])
                sims = cosine_similarity(q_vec, self.rubric_vectors).flatten()

                if target_company and target_company in self.company_profiles:
                    focus_keys = self.company_profiles[target_company].get("primary_focus", [])
                    for idx, key in enumerate(self.rubric_keys):
                        if key in focus_keys:
                            sims[idx] *= 1.5  # 50% priority boost for company-preferred topics!

                if not np.all(sims == 0):
                    best_idx = sims.argmax()
                    if sims[best_idx] > 0.05:
                        best_key = self.rubric_keys[best_idx]
                        rub = self.rubrics_data[best_key]
                        
                        rubric_text = (
                            f"[FAANG / DE SHAW EVALUATION RUBRIC ({rub.get('domain', '')})]:\n"
                            f"- Core Topic: {best_key}\n"
                            f"- Grading Standard: {' '.join(rub.get('evaluation_rubric', []))}\n"
                            f"- Probing Follow-Up Angle: {rub.get('follow_up_probe', '')}"
                        )
                        output_blocks.append(rubric_text)
            except Exception as e:
                print(f"[RAG] Rubric query error: {e}")

        return "\n\n".join(output_blocks)

