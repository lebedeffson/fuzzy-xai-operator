FROM python:3.10-slim
WORKDIR /app
COPY fuzzyxai_experiments/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY . /app
CMD ["bash", "run_all.sh"]
