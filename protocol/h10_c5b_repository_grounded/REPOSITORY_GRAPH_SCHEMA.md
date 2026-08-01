# Repository-grounded route graph

Node kinds:

`repository`, `package`, `module`, `file`, `class`, `function`,
`configuration_key`, `dependency`, `test`, `fixture`, `data_schema`,
`serialized_artifact`, `model_checkpoint`, `explainer`, and
`runtime_exception`.

Edge kinds:

`imports`, `calls`, `reads`, `writes`, `loads`, `serializes`, `depends_on`,
`configured_by`, `tested_by`, `fails_in`, `produces`, `consumes`, and
`explains`.

Every node and edge carries observable evidence references. Missing source or
runtime evidence produces `INSUFFICIENT_EVIDENCE`; it is never replaced by a
generic configuration guess.
