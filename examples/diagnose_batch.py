from fuzzyxai import FuzzyXAI


routes = (
    {"route_id": "empty", "nodes": [], "edges": []},
    {
        "route_id": "valid",
        "nodes": [
            {
                "node_id": "model",
                "node_type": "model",
                "registered_attributes": {"version": "v1"},
                "observed_attributes": {"version": "v1"},
            }
        ],
        "edges": [],
    },
)

batch = FuzzyXAI().diagnose_batch(routes=routes)
print(batch.route_status_counts)
