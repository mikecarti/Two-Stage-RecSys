# closest version to kaggle environment
FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app .

CMD ["fastapi", "run", "infer.py", "--port", "80"]