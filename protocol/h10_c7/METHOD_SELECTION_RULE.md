# H10-C7 method selection

Variants R0-R8 are evaluated only on open development repositories. A variant
is eligible only with zero Gold leakage, coverage at least 0.80, false
localization no worse than R0, Recall@20 at least 0.85, Recall@10 at least
0.75, contract macro-F1 at least 0.55 and a median context of at most 20
symbols.

The winner is selected lexicographically:

1. maximum Recall@10;
2. maximum contract macro-F1;
3. maximum joint file+symbol+contract Hit@3;
4. minimum context size;
5. minimum runtime;
6. stable variant identifier.

The runner writes `H10_C7_METHOD_LOCK.json` only when all gates pass. If no
variant passes, the status is `H10_C7_BLOCKED_DEVELOPMENT_GATE`, no held-out
manifest is created and the scientific result remains `NOT_EVALUATED`.
