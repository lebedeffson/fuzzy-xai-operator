# H10-C5c Decision Rules

`H10_C5C_PROTOCOL_AMENDMENT_001.json` supersedes the original minimum
development incident count and adds retrieval, contract and output-state
gates. It was created before H10-C5c development data collection or scoring.

H10-C5c is a prospective retrieval and contract-inference cycle. It does not
replace or rescore H10-C5b.

The 24 official H10-C5b held-out incidents may be used only for descriptive
post-hoc error analysis. Their repositories are forbidden in H10-C5c held-out.

Before any new held-out set is created, development must contain at least
30 incidents from at least eight repositories and satisfy every registered
development gate:

- candidate Recall@10 is at least 0.75;
- O_ROUTE joint file+symbol+contract Hit@3 is strictly greater than B_GREEDY;
- coverage is at least 0.70;
- false localization is not worse than B_GREEDY;
- contract accuracy is at least 0.60;
- O_ROUTE and B_GREEDY top-k outputs are structurally distinct;
- Gold leakage is zero.

Failure of this technical gate stops the cycle. Thresholds, endpoints and
repository exclusions must not be changed after development results are
observed.

Any later held-out set must contain at least 30 incidents from at least ten
repositories disjoint from development and from the official H10-C5b
held-out repositories. Its construction and one-time scoring require separate
locks not defined by this implementation-stage protocol.
