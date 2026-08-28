# Publication and dataset sources

## Main published line

1. Averkin A. N., Volkov E. N., Yarushev S. A. **Hybrid Method of Image
   Analysis Based on Artificial Intelligence Technologies and Fuzzy Sets**.
   *Journal of Computer and Systems Sciences International*, 2025, 64(3),
   460–473. English DOI: https://doi.org/10.1134/S1064230725700431.
   Russian DOI: https://doi.org/10.31857/S0002338825030103.

2. Volkov E. N., Averkin A. N. **Possibilities of Applying Grad-CAM in the
   Deep Learning-Based Diabetic Retinopathy Grading**. *Soft Measurements and
   Computing*, 2024. DOI:
   https://doi.org/10.36871/2618-9976.2024.12.007. The registered lineage is
   VGG-16 → five-grade DR classification → Grad-CAM.

The 2025 line mentions APTOS2019, FDGAR, IDRiD, EfficientNetB0, CenterNet,
ResUNet++ and CLAHE. Exact original weights, the complete training code and
all preprocessing parameters are not assumed available here. Canonical claim:
**«Воспроизводимое продолжение опубликованной постановки на открытых данных».**

## Supporting lineage

- **Hybrid Explainable Framework for Diabetic Retinopathy Classification from
  Fundus Images**, DOI https://doi.org/10.1109/SCM62608.2024.10554254.
- **Software Module for Fundus Image Analysis: A Multimodal Hybrid Approach
  with an Explainable Interface**, DOI
  https://doi.org/10.36871/2618-9976.2024.12.008.
- **Possibilities of Application of Neuro-Fuzzy Networks for Ophthalmologic
  Image Classification**, DOI https://doi.org/10.1134/S1054661824700421.
- **Possibilities of Explainable Artificial Intelligence for Glaucoma
  Detection Using the LIME Method as an Example**, DOI
  https://doi.org/10.1109/SCM58628.2023.10159038.

These works establish lineage only; their reported numbers are never copied as
new experiment results.

## Data sources

- APTOS 2019 Blindness Detection, official Kaggle competition data:
  https://www.kaggle.com/competitions/aptos2019-blindness-detection/data.
  Only the labeled training portion is allowed; hidden/private test labels are
  excluded. Access is subject to competition rules.
- IDRiD official Grand Challenge data page:
  https://idrid.grand-challenge.org/Data/ (links to IEEE DataPort). The page
  reports 516 grading images and 81 pixel-annotated images and identifies the
  dataset license as CC BY 4.0.
- IDRiD challenge analysis: Porwal et al., *Medical Image Analysis* 59 (2020),
  101561, DOI https://doi.org/10.1016/j.media.2019.101561.

Raw data are not redistributed by this repository.
