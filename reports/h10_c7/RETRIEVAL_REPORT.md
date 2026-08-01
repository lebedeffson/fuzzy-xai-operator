# H10-C7 retrieval implementation

The prospective implementation contains deterministic BM25, two-encoder dense
retrieval interfaces, reciprocal-rank fusion, a personalized repository graph,
a fixed structural reranker and a bounded repository explorer. Transformer
backends are local-only and pinned by repository revision. No model result is
reported until local weight hashes are recorded and new development evidence
is supplied.

The old H10-C5c retriever remains R0 and is not changed.
