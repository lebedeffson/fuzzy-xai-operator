# External validation release gate

The computational framework cannot manufacture evidence that requires independent people.

## Comprehension pilot

Run `python scripts/build_external_validation_package.py` to freeze the three scenarios, their hashes, protocol,
response schema, scorer, and limitations. At least six external participants must complete both conditions for all
scenarios. The repository stores only anonymized records. The release gate accepts only a scorer result of `pass`.

## Domain-language review

An independent subject-matter expert must review the exact domain dictionary hash and complete
`release_evidence/domain_language_review/review_record.json`. Project authors cannot self-approve it.

## Verification

```bash
python scripts/verify_external_release_gates.py
python scripts/verify_external_release_gates.py --require-pass
```

The first command validates package integrity and may report `BLOCKED`. The second exits non-zero unless both real
external gates pass. No `v1.2.0` tag, demonstrated-comprehensibility claim, or categorical regulated-domain wording
is allowed while either gate is open.
