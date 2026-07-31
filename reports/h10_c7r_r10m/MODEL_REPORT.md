# H10-C7R R10M model report

Both neural components used frozen upstream weights without fine-tuning. Inference was local-only after snapshot acquisition.

- `microsoft/graphcodebert-base` revision `2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d`, snapshot SHA256 `22713ebac7355505dc28acf6992d6092ecd31b6f9bfb87b967d53453400d9eea`, weights SHA256 `fc542850abf74be2df516bcdedfc2dcdb9bd02c8098a6d5f4d63da73cbcb9e71`.
- `BAAI/bge-reranker-v2-m3` revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, snapshot SHA256 `505e20b19c5937824d54b3b91e2b64f3f023413d183e8d18ccdbd0f51f74197a`, weights SHA256 `d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286`.

Runtime: `{"device": "NVIDIA GeForce RTX 4060 Laptop GPU", "huggingface_hub": "1.18.0", "python": "3.14.6", "sentence_transformers": "NOT_INSTALLED_NOT_USED", "torch": "2.11.0+cu128", "transformers": "5.10.2"}`.
