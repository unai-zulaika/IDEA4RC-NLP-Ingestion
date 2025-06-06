FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONPATH="${PYTHONPATH}:/app"

# set environment variable for model password
ENV MODEL_PASSWORD=your_model_password_here


CMD ["uvicorn", "app:app", "--port", "8000","--host", "0.0.0.0"]
