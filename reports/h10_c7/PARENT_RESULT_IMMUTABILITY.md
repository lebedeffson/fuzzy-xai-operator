# H10-C7 parent-result immutability

H10-C7 starts from parent release commit
`1fed9ac074295bc4b6cad33841b73a4022bcafcd`. It does not rescore, replace or
reinterpret H10-C5b, H10-C5c, H10-C6-N or H9-E2E-v2.

The protocol lock records these published parent artifact hashes:

| Parent result | SHA256 |
| --- | --- |
| H10-C5b | `a1ac841880e9acac2c09eac79d18721d17f83da5e8cec19a7057fa8cb453fc6d` |
| H10-C5c | `60810d8ee58b20c6d9b1a5e4800c23fc678ce9e698ad70d339e91d37a7f69ba1` |
| H10-C6-N | `a3fc46e01c7fec06b539cb880d749f047c1f49fc218706c17e5d04a62bc8f441` |
| H9-E2E-v2 | `0d6a5686043ff0c158b6b0d9700e595b9f4398ac4c549431f361fd1b1dc0d6a8` |

The canonical operator manifest remains unchanged with SHA256
`479fa678b95d5e8c4334136f84d619735049e7f9009c6a7f9951e1ca741ef73a`.
H10-C7 operator mappings are isolated in
`framework/fuzzyxai/operators_manifest_h10_c7_addendum.yaml`.

Verification performed after the H10-C7 implementation:

- recorded parent artifact hashes: `PASS`;
- canonical operator manifest SHA256: `PASS`;
- compatible H10-C5b/H10-C5c/H9 test subset: `126 passed`;
- parent results modified by H10-C7: `false`.
