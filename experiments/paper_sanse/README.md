# SANSE: unweighted taxonomy reconstruction from manuscript Figure 3

This is a **secondary transcription of the author's published image**, not
the original literature-screening database or independent annotation.
Every printed line in the 3 by 5 grid is counted as one configuration
even when it groups related algorithms (GAM/EBM, LRP/DeepLIFT, PDP/ICE/ALE).
Original embedded image: word/media/image3.png extracted from the
manuscript; SHA256:
7015f76b602fc13646e2dc6e35bc08cb0127beee274c9a3973d569462fb63106.

Inputs:
- figure3_transcription_47.csv: 47 source/object/audience triples,
  source names, and explicit figure-cell locator.
- recheck_figure3.py: Python standard library only. It validates 47
  entries and all 15 figure cells; recomputes entropy, mutual information,
  collision counts, leave-one-out MI ranges, group retrieval size,
  third-axis negative control and tie-rule sensitivity.

## Verifiable unweighted values

| Statistic | Recomputed | Article |
| --- | ---: | ---: |
| Entropy of source/object/audience, bits | 1.406397 / 2.115218 / 2.068600 | 1.406 / 2.115 / 2.069 |
| MI source-object | 0.1633049 | 0.1633 |
| MI source-audience | 0.3771108 | 0.3771 |
| MI object-audience | 0.6081598 | 0.6082 |
| Same-category pairs source/object/audience | 442 / 259 / 269 | 442 / 259 / 269 |
| Same-category pairs source+object / object+audience / three axes | 119 / 80 / 45 | 119 / 80 / 45 |
| Mean candidate group (object+audience / all three) | 4.4043 / 2.9149 | 4.40 / 2.91 |
| Joint entropy source+object / all three | 3.358309 / 4.291098 | 3.358 / 4.291 |

All reported leave-one-out mutual-information ranges are reproduced to
the printed four decimal places.

Third-axis negative control: 50,000 fixed-marginal permutations of
audience labels (new seed 20261130). Choose the best two axes anew for
every permutation. Observed gain 35 indistinguishable pairs; null mean
34.85036; one-sided p=(25828+1)/(50000+1)=0.51656967. Original
article's 34.861 and p=0.51795 agree within Monte Carlo noise. This
does NOT demonstrate a meaningful benefit from the audience axis.

Leave-one-out prediction of audience from source/object is tie-rule
sensitive: abstain on local ties gives 15/47, alphabetical choice
17/47, global-frequency tie-break 23/47. The global majority baseline
gives 18/47. The archival algorithm did not specify the tie rule, so
its 15-vs-18 ranking cannot be treated as a stable scientific result.

## Independent computational check, not independent annotation

[GitHub Actions 43396444400](https://github.com/lebedeffson/fuzzy-xai-operator/actions/runs/43396444400):
Python 3.11.18 and 3.12.16 both SUCCESS. Workflow artifacts
14023461007 and 14023182353 were examined. All counts, negative
control values, and tie-rule outcomes agree. Some floating-point MI
values differ by no more than 8.9e-16 due to order of summation,
far below displayed precision.

## What remains unavailable

- Original per-method counts of 355 literature mentions, hence no
  reproduction of mention-weighted MI=0.3512.
- Primary bibliographic sources and full selection trace 8173 to 47.
- Independent method-label audits, original RNG seed, and proof that
  category assignment benefits unseen users.
- The printed Figure 3 itself derives from the authors' labels; our
  reproducibility check is CONDITIONAL on their correct labeling.

This PR changes no ML runtime code, original data, scored dissertation
artifacts, or locked manuscript files.
