# Anonymization procedure

1. Replace participant and reviewer identifiers with one-way study hashes before analysis.
2. Store consent records separately from response records.
3. Remove free-text identifiers and operational metadata not required by the preregistered analysis.
4. Validate the response schema and duplicate keys before scoring.
5. Publish only anonymized records and aggregate results permitted by the ethics determination.

The scorer never creates human responses and rejects records that do not satisfy the frozen design.
