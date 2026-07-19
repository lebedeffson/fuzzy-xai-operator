FROM python:3.11-slim
WORKDIR /app
COPY requirements.lock pyproject.toml README.md /app/
COPY framework /app/framework
RUN python -m pip install --no-cache-dir -r requirements.lock && \
    python -m pip install --no-cache-dir --no-deps .
COPY . /app
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
CMD ["make", "reproduce-dissertation"]
