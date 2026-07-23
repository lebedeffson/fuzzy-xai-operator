from fuzzyxai import FuzzyXAI

from diagnose_single_route import route


report = FuzzyXAI().diagnose(route=route)
for audience in ("user", "expert", "audit"):
    print(f"\n[{audience}]\n{report.summary(audience)}")
