# closest version to kaggle environment
FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app .

#CMD ["uvicorn", "app:app.py", "--host", "0.0.0.0", "--port", "80"]
CMD ["fastapi", "dev", "app.py", "--port", "80", "--host", "0.0.0.0"]