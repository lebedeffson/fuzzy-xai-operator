from fuzzyxai import FuzzyXAI

from diagnose_single_route import route


report = FuzzyXAI().diagnose(route=route, repair_mode="plan")
for step in report.repair_plan.steps if report.repair_plan else ():
    print(step.step_id, step.operation, step.preconditions, step.verification_checks)
