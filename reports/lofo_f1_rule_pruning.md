# LOFO-F1 rule pruning demo

Method: `LOFO-F1 / Leave-One-Rule-Out F1 importance`
Formula: `z_without_r = z_full - H_val[:, r] * theta[r]; importance_r = F1_full - F1_without_r`

## Quality

- Full F1: `0.989487`
- LOFO-F1 top-B F1: `0.989461`
- Budget-Prune top-B F1: `0.984069`
- Bootstrap LOFO-F1 top-B F1: `0.989487`

## Stability proxy

- Jaccard LOFO vs bootstrap LOFO: `0.846154`
- Jaccard LOFO vs Budget-Prune: `0.411765`

## Top LOFO-F1 rules

- 1. `r_000`: delta_f1=`0.1599`, f1_without=`0.829587`
- 2. `r_002`: delta_f1=`0.093596`, f1_without=`0.89589`
- 3. `r_003`: delta_f1=`0.070532`, f1_without=`0.918955`
- 4. `r_005`: delta_f1=`0.057036`, f1_without=`0.93245`
- 5. `r_001`: delta_f1=`0.039575`, f1_without=`0.949912`
- 6. `r_004`: delta_f1=`0.032242`, f1_without=`0.957245`
- 7. `r_012`: delta_f1=`0.001222`, f1_without=`0.988264`
- 8. `r_026`: delta_f1=`0.001222`, f1_without=`0.988264`
- 9. `r_070`: delta_f1=`0.001222`, f1_without=`0.988264`
- 10. `r_085`: delta_f1=`0.001208`, f1_without=`0.988279`

## Conclusion

LOFO-F1 ranks rules by real validation F1 drop without retraining; bootstrap aggregation turns it into a stable top-B selector.
