# Docker Compose smoke

- Status: `PASS`
- Image: `sha256:8d17b573beec919496fb4a07e2845d74e9af9d955057db09a30e246ef5275301`
- API `/ready`: `PASS`
- UI title: `FuzzyXAI ML Vertical v1`
- MLflow `/health`: `OK`
- API `/explain`: `ACCEPT / F0 / valid`
- MLflow logging through API: `MLFLOW_INTEGRATION_PASS`
- Required artifacts logged: `9`

The smoke used isolated host ports `18000`, `18092`, and `15001`; service ports and the default one-command Compose interface remain stable.
