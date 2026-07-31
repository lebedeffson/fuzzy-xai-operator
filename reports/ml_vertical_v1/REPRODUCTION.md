# Reproduction

```bash
pip install -e '.[dev,ml-vertical]'
make ml-vertical-test
make ml-vertical-acceptance
docker compose up --build
```
