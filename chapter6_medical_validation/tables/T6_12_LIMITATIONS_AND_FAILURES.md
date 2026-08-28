| domain | case | what_happened | detected | not_detected | interpretation |
| --- | --- | --- | --- | --- | --- |
| ECG | ECG_D | confident false positive received technical accept | internally consistent route only | semantic prediction error | without external verification evidence, route consistency is not ground-truth correctness |
| ECG | ECG_E | confident false negative received technical accept | internally consistent route only | semantic prediction error | without external verification evidence, route consistency is not ground-truth correctness |
| ECG | ECG_I | controlled checkpoint mismatch | critical integrity fault | not applicable | critical override blocks the action |
| ECG | ECG_H | controlled provenance fault | critical integrity fault | not applicable | critical override blocks the action |
| brain_v1_pilot | all cases | single-atlas validation with seven held-out patches | limitation recorded in provenance/report | cross-atlas generalization | pilot metrics do not establish broad neuroanatomical transfer |
| brain_v2_confirmatory | all cases | single-atlas section-block validation | controlled integrity faults only | cross-atlas or clinical generalization | v2 increases anatomical sampling coverage but remains one-atlas evidence |
