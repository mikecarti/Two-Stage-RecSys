# closest version to kaggle environment
FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
VOLUME /app/model_files

COPY app .

CMD ["fastapi", "dev", "infer.py", "--port", "80"]