# Third-party notices

The FuzzyXAI source code is distributed under the repository `LICENSE` (MIT).
Third-party datasets, model weights, and libraries keep their own terms and are
not relicensed by this repository.

## Chapter 4 v13 contour

| Resource | Pinned source | Recorded license status | Release handling |
|---|---|---|---|
| AG News (`fancyzhx/ag_news`) | revision `eb185aade064a813bc0b7f42de02595523103ca4` | The pinned dataset card declares `license: unknown`. | Raw/processed text and labels are excluded. The reproducer downloads the source directly. |
| DistilBERT AG News (`textattack/distilbert-base-uncased-ag-news`) | revision `52ee64de95f38323f136c6f6b05e1af7c433417e` | The pinned model card does not state a license for the weights. | Weights and cache files are excluded. The reproducer downloads the source directly. |

The released evidence contains only derived aggregate measurements and sanitized
rows without source text, tokens, labels, or model weights. Users must verify
upstream terms before downloading or redistributing either resource.

Python package versions used by the contour are pinned in
`config/chapter4_v13_requirements.txt`; their respective package licenses apply.
