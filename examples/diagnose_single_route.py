from fuzzyxai import FuzzyXAI


route = {
    "route_id": "example:single",
    "nodes": [
        {
            "node_id": "preprocessor",
            "node_type": "preprocessing",
            "registered_attributes": {"version": "v1"},
            "observed_attributes": {"version": "v2"},
            "evidence_refs": ["manifest:preprocessor"],
        }
    ],
    "edges": [],
}

report = FuzzyXAI().diagnose(route=route, repair_mode="plan")
print(report.summary("user"))
