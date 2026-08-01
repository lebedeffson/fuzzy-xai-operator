# H10-C7-R5C error analysis

False confirmations:

- Selected C0: none, because it failed closed.
- Rejected C1: seven out-of-fold confirmations, all false. This demonstrates
  that its training-fold threshold did not transfer safely across excluded
  repositories.
