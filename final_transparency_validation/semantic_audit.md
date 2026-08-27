# P19 semantic audit

Проверка основана на public-runtime JSON, формулах и реальных graph nodes, а
не только на результате pytest.

## System accept

| Quantity | Value/status | Source and actual inputs | Definition/policy | Provenance | Verdict |
| --- | --- | --- | --- | --- | --- |
| Gamma | 0, measured/certified | generic RF provider -> `SystemSourceEvidence` -> `class-probability-to-risk-partition-v2`; transformed low=1, target low=1 | `d_E(T_ij(E_model),E_target)`, beta 0.30/0.25/0.15/0.20/0.10 | `E_model -> T_ij -> aligned`, aligned+target -> Gamma | verified; `d_L=1/3` is diagnostic-only and absent from beta |
| U_model | 0, measured | standard deviation of 200 binary per-tree predictions | `ensemble_vote_standard_deviation`, expected range [0, 0.5] | `U_model` | verified; this is not variance |
| U_rules | 0, measured | full risk-rule coverage, no conflict | coverage/conflict policy | `U_rules` | verified |
| U_trace | 0, measured | required fields present and externally reported complete | required-field check plus `externally_verified_trace_status`; verifier source retained | `U_trace` | verified as externally supplied status, not independent runtime verification |
| u_M | 0 | 0.5*0 + 0.3*0 + 0.2*0 | ExplainPlan eta | three uncertainty nodes -> `u_M` | verified |
| Delta | 0, measured | interval [0,0], midpoint 0, inverse [0,0], D_F=0 | `D_F(F_int,iota(Pi(F_int)))` | representation -> reduction -> Delta | verified, lossless degenerate interval |
| I_pre | 0.8962481423 | H=.0143565, C=.3333333, O=.2, K=0, U=0; weights .2 each | `exp(-L(E_pre))` | E_pre -> I_pre | verified |
| rho | 0.0207503715, complete | .3*0 + .25*0 + .2*.1037518577 + .15*0 + .1*0 | strict five-term formula | rho_p/u_M/(1-I_pre)/Delta/chi_R -> rho | verified; Gamma is diagnostic ancestry, not a direct rho term |
| action | accept | candidate=accept from rho; critical_override=false | threshold policy + override resolution | rho -> candidate; chi_R -> override; both -> action | verified |

## System conflict

Это контролируемая fault-injection: target trace намеренно имеет
`source=missing-checkpoint`, `checksum=missing`.

| Quantity | Value/status | Source and actual inputs | Definition/policy | Provenance | Verdict |
| --- | --- | --- | --- | --- | --- |
| Gamma | 0.05, measured/certified | d_tau=.5; other beta components 0 | trace weight .1; shared gamma_max=.60 | transform comparison -> Gamma | verified; accept/conflict use the same ExplainPlan threshold |
| U_model/U_rules/U_trace | 0 / 0 / 1 | native votes, rule coverage, controlled externally reported trace fault | independent typed channels | three nodes -> u_M | verified |
| u_M | 0.2 | .5*0 + .3*0 + .2*1 | ExplainPlan eta | u_M | verified |
| Delta | 0 | [0,0] -> 0 -> [0,0], D_F=0 | real interval reduction | Delta | verified |
| I_pre | 0.8611057500 | same H/C/O/K, U=.2 | `exp(-L(E_pre))` | E_pre -> I_pre | verified |
| rho | 0.1777788500, complete | .3*0 + .25*.2 + .2*.13889425 + .15*0 + .1*1 | strict five-term formula | rho | verified |
| action | block | numeric candidate=accept; critical_override=true | fail-closed critical policy | rho -> candidate; chi_R -> override; both -> action | verified |

## System reduction / uncertainty

The public runtime selected held-out object 74 because its real 200-tree vote
distribution is non-degenerate; no probability or Delta was injected.

| Quantity | Value/status | Source and actual inputs | Definition/policy | Provenance | Verdict |
| --- | --- | --- | --- | --- | --- |
| Gamma | 0.3334630325, certified | executed T_ij, d_R=.666667, d_alpha=.603889, d_u=.381065 | weighted d_E after transform | E_model -> T_ij -> Gamma | verified |
| U_model | 0.4987734957 | votes 107/93, proportions .535/.465 | standard deviation of binary per-tree predictions, range [0,.5] | U_model | verified; not variance |
| U_rules | 0.1177083333 | coverage=.117708, conflict=0 | max(conflict, 1-max activation) | U_rules | verified |
| U_trace | 0 | required trace complete | trace policy | U_trace | verified |
| u_M | 0.2846992478 | .5*.4987735 + .3*.1177083 + .2*0 | ExplainPlan eta | three channels -> u_M | verified |
| Delta | 0.4817304978 | [.0365390,1] -> .5182695 -> [.5182695,.5182695] | D_F after Pi/iota | representation -> reduction -> Delta | verified positive information loss |
| I_pre | 0.8077111390 | H=.2497213, C=.3333333, O=.2, K=0, U=.2846992 | exp(-L(E_pre)) | E_pre -> I_pre | verified |
| rho | 0.3424859088 | .3*.5353125 + .25*.2846992 + .2*.1922889 + .15*.4817305 + .1*0 | strict five-term formula | five numeric inputs -> rho | verified |
| action | accept | numeric candidate=accept; no override | threshold + override resolution | candidate + override -> action | verified |

## Training

- Actual run: `sgd-bcw-p19-run-19`, `SGDClassifier.partial_fit(log_loss)`,
  30 epochs, final checkpoint `epoch:30`.
- Final model fingerprint:
  `510386e38ad7319ce15399026dec514fbdcef74a8b3a954e4cec79b15c3a7e68`.
- First learned epoch 1; forgetting at epoch 27 is a real
  `correctness_transition` (true -> false), confidence 0.527447 -> 0.525016.
- Stability 0.965517; loss is measured, not substituted by zero.

## Image Integrated Gradients

- Target class 0, all-zero tensor baseline, logit space.
- F_target(x)=8.5601491928; F_target(baseline)=-1.3766483068.
- Input-output delta=9.9367974997. At 512 trapezoidal intervals the attribution
  sum is 9.9352016449.
- Completeness residual=0.0015958548; relative error=0.0001606005.
- The 16/32/64/128/256/512 public-API convergence audit is stored in
  `golden_image_28x28/ig_convergence.json`. The prior 28.65% residual came from
  changing the target class along the integration path; the runtime now fixes
  the source prediction target for every interpolation point.

## Local tabular risk contract

The local LogisticRegression route lacks the non-zero-weight uncertainty and
reduction inputs of the full dissertation rho. It therefore exports
`rho=None`, `status=incomplete`, missing `uncertainty` and `reduction_loss`,
and a separately named `partial_risk_score=0.104253`.
