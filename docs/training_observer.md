# Training observer

`FuzzyXAI.observe_training()` consumes measured epoch records. A record may include predicted class, confidence, loss, correctness, embedding, and rule activations.

Forgetting is a correct-to-wrong transition or a configured confidence collapse after learning. Subgroup averaging requires global improvement together with subgroup degradation, rule disappearance, or measured embedding collapse.

Run `examples/object_85_training_trace.py` for the controlled protocol. It trains an SGD classifier, removes a rare subtype from later epochs, measures object forgetting, restores the subtype, and reports the trade-off on overall accuracy and subgroup recall.
