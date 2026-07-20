# AI-assisted pre-review study

This directory contains blind inputs and immutable contracts for an AI pre-review followed by independent human confirmation.

The AI stage is a formative and confirmatory pre-review only. It is not domain approval, demonstrated comprehension, expert validation, or a replacement for human experts.

## Lifecycle

1. Build the 360-case, 1080-variant blind log.
2. Run independent external AI review sessions using the generated batches.
3. Import raw responses without rewriting them.
4. Complete the formative repair cycle and lock the confirmatory protocol.
5. Commit three AI confirmatory runs before distributing human packets.
6. Import signed responses from at least three independent human experts.
7. Compare AI scores with human consensus using the preregistered thresholds.

The method identity key is encrypted. Its password is stored outside the repository and is not included in analysis archives.
