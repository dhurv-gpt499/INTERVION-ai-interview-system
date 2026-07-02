def decide_categories(resume_parsed: dict, target_level: str, preferred_companies: list = None) -> list[str]:
    if preferred_companies is None:
        preferred_companies = []

    categories = {
        "dsa"                : 0.0,
        "projects"           : 0.0,
        "core_cs"            : 0.0,
        "skills_depth"       : 0.0,
        "system_design"      : 0.0,
        "behavioural"        : 0.0,
        "quant_brainteasers" : 0.0,
        "os_concurrency"     : 0.0,
        "dbms_sql"           : 0.0,
    }

    # Check for Quant & Finance companies
    quant_firms = ["de shaw", "citadel", "two sigma", "jane street", "jump trading", "tower research", "arcesium", "worldquant", "trexquant", "goldman sachs", "morgan stanley"]
    is_quant = any(any(q in comp.lower() for q in quant_firms) for comp in preferred_companies)
    
    # Check for FAANG / Big Tech
    faang_firms = ["google", "meta", "apple", "amazon", "microsoft", "netflix", "nvidia", "uber"]
    is_faang = any(any(f in comp.lower() for f in faang_firms) for comp in preferred_companies)

    # 1. Company-Specific Topic Boosting (The Anti-Project-First Protocol)
    if is_quant:
        categories["quant_brainteasers"] = 4.0  # Top priority for Quant firms!
        categories["os_concurrency"]     = 3.8
        categories["dsa"]                = 3.5
        categories["projects"]           = 2.5  # Projects tested later
    elif is_faang:
        categories["dsa"]           = 4.0       # FAANG prioritizes DSA & Complexity!
        categories["system_design"] = 3.5
        categories["projects"]      = 3.0
    else:
        # Standard weighting
        categories["projects"] = 3.0
        categories["dsa"]      = 2.5

    # 2. Candidate Background Enhancements
    if resume_parsed.get("competitive"):
        categories["dsa"] += 0.5

    if target_level in ("entry", "intern"):
        categories["core_cs"] = 2.5
        categories["dbms_sql"] = 2.0
    elif target_level in ("mid", "senior"):
        categories["system_design"] += 1.5

    if resume_parsed.get("skills"):
        categories["skills_depth"] = 2.0

    categories["behavioural"] = 1.0

    # Sort by weight descending, drop zeros, return ordered list
    return [
        topic for topic, weight in
        sorted(categories.items(), key=lambda x: x[1], reverse=True)
        if weight > 0
    ]