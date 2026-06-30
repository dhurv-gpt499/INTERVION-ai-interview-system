def decide_categories(resume_parsed, target_level):
    categories = {
        "dsa"           : 0,
        "projects"      : 0,
        "core_cs"       : 0,
        "skills_depth"  : 0,
        "system_design" : 0,
        "behavioural"   : 0,
    }

    # DSA
    if resume_parsed.get("competitive"):
        categories["dsa"] = 3
    else:
        categories["dsa"] = 2

    # Projects
    if resume_parsed.get("projects"):
        categories["projects"] = 3

    # Core CS
    if target_level in ("entry", "intern"):
        categories["core_cs"] = 2

    # Skills depth
    if resume_parsed.get("skills"):
        categories["skills_depth"] = 2

    # System design
    if target_level in ("mid", "senior"):
        categories["system_design"] = 2
    elif resume_parsed.get("experience"):
        categories["system_design"] = 1
    else:
        categories["system_design"] = 0

    # Behavioural
    categories["behavioural"] = 1

    # sort by weight, drop zeros, return ordered list
    return [
        topic for topic, weight in
        sorted(categories.items(), key=lambda x: x[1], reverse=True)
        if weight > 0
    ]