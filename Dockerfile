# Образ инференса; модель и веса (serving/) не копируются в образ, а монтируются:
#   docker build -t recsys .
#   docker run -p 8000:8000 -v "$PWD/serving:/app/serving:ro" recsys
# Не проверялся (на машине сборки нет Docker); проверенный путь — ./setup.sh.
FROM python:3.9-slim
WORKDIR /app
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY recsys recsys
COPY configs configs
COPY examples examples
EXPOSE 8000
CMD ["python", "-m", "recsys.inference", "serve", "--bundle", "serving", "--port", "8000"]
