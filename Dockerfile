FROM python:3.12-slim

# ffmpeg converts browser voice notes (webm/m4a) to WAV for Gemini
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY static ./static

ENV PYTHONUNBUFFERED=1 PORT=8080
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 180 app.main:app
