# FeatureForge - CPU-only, 零 GPU 依赖
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 默认执行基准（落盘 /app/benchmark.json）
CMD ["python", "-m", "featureforge.examples.run_demo"]
