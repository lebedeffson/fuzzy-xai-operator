# AI pre-review and independent human confirmation

This protocol tests whether a language-model pre-review can identify defects in model explanations before an expensive independent expert review. It does not treat AI output as expert evidence.

The technical pipeline freezes 360 unique cases across tabular, image, text and time-series modalities, with 240 formative and 120 confirmatory cases. Each case has three independently blinded explanation variants. Raw frozen benchmark evidence remains separate from explicitly labeled controlled route conditions.

Confirmatory locking is fail-closed. It requires a real accepted formative run. Human packet generation additionally requires three imported confirmatory AI runs and an AI score commitment. Human results cannot be synthesized by repository code.

Until those external stages are complete, claims remain `planned_not_run`, `pending_three_ai_runs`, or `external_gate`, and stable release remains blocked.
