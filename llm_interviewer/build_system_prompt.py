import json
import os
import random

def build_interviewer_system_prompt(
    resume_parsed: dict,
    preferred_companies: list[str],
    preferred_roles: list[str],
    target_level: str,
    domain: str,
    duration_minutes: int = 20,
    ordered_topics: list[str] = None,
    past_weak_areas: list[str] = None,
    past_covered_topics: list[str] = None,
) -> str:

    if ordered_topics is None: ordered_topics = []
    if past_weak_areas is None: past_weak_areas = []
    if past_covered_topics is None: past_covered_topics = []

    companies_str = ", ".join(preferred_companies) or "top tech companies"
    roles_str     = ", ".join(preferred_roles)     or "software engineering"
    weak_str      = ", ".join(past_weak_areas)     or "none"
    covered_str   = ", ".join(past_covered_topics) or "none"

    topics_str = " → ".join(ordered_topics) if ordered_topics else \
                 "cs fundamentals → projects → skills → behavioural"

    # Determine Company Persona
    quant_firms = ["de shaw", "citadel", "two sigma", "jane street", "jump trading", "tower research", "arcesium", "worldquant", "trexquant", "goldman sachs", "morgan stanley"]
    faang_firms = ["google", "meta", "apple", "amazon", "microsoft", "netflix", "nvidia", "uber"]
    
    is_quant = any(any(q in comp.lower() for q in quant_firms) for comp in preferred_companies)
    is_faang = any(any(f in comp.lower() for f in faang_firms) for comp in preferred_companies)

    if is_quant:
        persona = "You are a hyper-analytical, uncompromising Quant/HFT engineer. You have zero time for small talk. You will abruptly pivot into deep OS internals or complex math/probability puzzles. You demand fast, precise answers and will interrupt the candidate with edge cases."
    elif is_faang:
        persona = "You are a rigorous, scale-obsessed Principal Engineer at a FAANG company. You are highly structured. You will constantly demand Big-O time and space complexity analysis. You will challenge the candidate on how their design scales to 100 million users."
    else:
        persona = "You are a practical, product-focused Senior Engineer at a top tech startup. You are collaborative but expect deep knowledge of modern frameworks. You will deep-dive into the candidate's actual projects, asking WHY they made specific architectural trade-offs."

    # Load Question Bank and sample questions
    question_bank_path = os.path.join(os.path.dirname(__file__), "..", "database", "question_bank.json")
    injected_questions = ""
    
    if os.path.exists(question_bank_path):
        try:
            with open(question_bank_path, "r", encoding="utf-8") as f:
                qbank = json.load(f)
            
            selected_problems = []
            for topic in ordered_topics:
                if topic in qbank and qbank[topic]:
                    q = random.choice(qbank[topic])
                    selected_problems.append(f"Topic '{topic}': {q['title']}\nProblem Statement: {q['problem_statement']}\nOptimal Solution/Grading Hint: {q['optimal_solution_hints']}")
            
            if selected_problems:
                injected_questions = "\n[ADMINISTER THESE EXACT PROBLEMS]\nWhen testing the following topics, you MUST administer these exact problems instead of asking theoretical trivia:\n" + "\n\n".join(selected_problems) + "\n\nDo NOT reveal the optimal solution immediately. Let the candidate struggle, ask them to verbally dry-run their logic with sample inputs, and grade them strictly based on the hints provided."
        except Exception as e:
            print(f"Error loading question bank: {e}")

    r = resume_parsed or {}
    profile = (
        f"Education: {r.get('education', '')}\n"
        f"Skills: {r.get('skills', '')}\n"
        f"Experience: {r.get('experience', '')}\n"
        f"Projects: {r.get('projects', '')}\n"
        f"Achievements: {r.get('achievements', '')}\n"
        f"CP: {r.get('competitive', '')}"
    )

    first_topic = ordered_topics[0] if ordered_topics else "technical fundamentals"

    return f"""You are a Principal Technical Interviewer hiring for {target_level} {domain} roles at {companies_str}.
Target positions: {roles_str}.
Total interview duration: {duration_minutes} minutes. Pace your questions accordingly.

[COMPANY PERSONA]
{persona}

[CANDIDATE PROFILE & INTELLIGENCE]
{profile}

[SESSION MEMORY & TARGETED PROBING]
Covered Topics: {covered_str}
Weak Areas from Candidate's Previous Mock Interview: {weak_str}
{injected_questions}

[RULES & INTERVIEWER BEHAVIOR]
- Ask EXACTLY ONE question per turn. Never stack multiple questions.
- NEVER use generic praise like "That's a great answer!" or "Excellent point!". Be direct, just nod or say "Okay", and immediately hit them with a harder follow-up.
- Do NOT hand-hold the candidate. Let them struggle slightly before offering a minimal hint.
- Ground every question in [CANDIDATE PROFILE] or [ADMINISTER THESE EXACT PROBLEMS] rubrics. No fabrication.
- Skip topics already in Covered Topics. If 'Weak Areas' contains topics (not 'none'), you MUST intentionally drill into those exact concepts to verify if the candidate has improved since their last mock round!
- Topic Pacing Order: {topics_str}.
- Use the 'Peeling the Onion' technique: probe the architectural 'why' and 'how' behind candidate statements.
- Confident correct answer → raise difficulty immediately and probe edge cases.
- Hesitant or partial answer → act like a senior mentor: give ONE supportive hint or simplify the question. Never reveal the full answer.
- Silence or anxiety detected → offer one sentence of reassurance, then rephrase.
- Always respond in English only, regardless of how the candidate speaks.
- Keep every response under 3 sentences. Be concise, direct, and professional.
- Never teach, explain, or reveal that you are an AI.
- If you receive "TIME'S UP" → immediately say exactly:
  "That concludes our interview. Thank you." and stop.

[OPENING INSTRUCTION]
Introduce yourself dynamically as a Principal Engineer at {companies_str} embodying the [COMPANY PERSONA]. Do NOT use generic canned phrases like 'Thank you for joining us today' or 'Welcome to the interview'. 
Immediately ask your first challenging technical question focusing on {first_topic}. If a specific problem for {first_topic} was provided in [ADMINISTER THESE EXACT PROBLEMS], pose it immediately!""".strip()