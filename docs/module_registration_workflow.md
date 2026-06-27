# Module Registration Workflow

1. Create a registry entry from `templates/module_registry_entry.json`.
2. Set `registry_id`, `source_repo`, `source_artifact`, `adapter`, `status`, `evidence_level`, and `claim_scope`.
3. Add or intentionally omit fixture data. Missing fixtures produce warnings, not fake metrics.
4. Validate metadata:

```bash
python scripts/register_external_module.py --module-id my_module --entry-json path/to/entry.json
```

5. Implement an adapter by copying `templates/adapter_stub.py` and subclassing `fuzzyxai.sdk.BaseAdapter`.
6. Rebuild evidence:

```bash
make ecosystem-evidence
make dissertation-artifacts
```

The workflow validates interface compatibility and evidence boundaries. It does not certify external model quality.
