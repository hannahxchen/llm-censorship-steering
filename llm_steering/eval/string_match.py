import re
from ..data import TARGET_PATTERNS

def get_refusal_counts(outputs):
    counts = {"refusal": 0, "partial_refusal": 0, "compliance": 0}

    for x in outputs:
        if re.search(TARGET_PATTERNS["refusal"], x):
            counts["refusal"] += 1

        elif re.search(TARGET_PATTERNS["partial_refusal"], x):
            counts["partial_refusal"] += 1

        elif re.search(TARGET_PATTERNS["compliance"], x):
            counts["compliance"] += 1
    

    percentage = {target: count if count != 0 else count for target, count in counts.items()}
    return percentage