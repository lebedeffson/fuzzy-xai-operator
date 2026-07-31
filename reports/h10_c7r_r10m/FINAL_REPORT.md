# H10-C7R R10M final development report

Status: `H10_C7R_R10M_DEVELOPMENT_NOT_SUPPORTED`.

Scientific result: `NOT_EVALUATED`.

| Method | File R@10 | File R@20 | Pool R@200 | Symbol R@20 | MRR | False localization | Runtime, s |
|---|---:|---:|---:|---:|---:|---:|---:|
| B_BM25 | 0.5500 | 0.5500 | 0.4500 | 0.4500 | 0.1411 | 0.5500 | 2.087 |
| B_TRACE | 0.1000 | 0.1000 | 0.0750 | 0.0750 | 0.0208 | 0.9250 | 0.000 |
| R10A | 0.8250 | 0.8250 | 0.4000 | 0.4000 | 0.0992 | 0.6000 | 4.617 |
| R10M | 0.9500 | 0.9500 | 0.8750 | 0.5250 | 0.1329 | 0.4750 | 16.083 |
| R9 | 0.8750 | 0.8750 | 0.5500 | 0.5500 | 0.2032 | 0.4500 | 12.884 |

The frozen development gate did not pass. Coverage and file Recall@10 passed, while file Recall@20, pool Recall@200, symbol Recall@20, repository Q1, baseline superiority, and MRR superiority did not all pass. In accordance with the protocol, no confirmatory held-out was created and no further R10 variant is opened.

Observable leakage audit: `PASS`, Gold leakage `0`.
