FROM node:24-bookworm-slim AS frontend
WORKDIR /build
COPY frontend/package*.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
COPY requirements-lock.txt requirements-voiceguard.txt requirements-voiceguard-transitive.txt ./
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -c requirements-voiceguard-transitive.txt -r requirements-voiceguard.txt
COPY callguard/ ./callguard/
COPY voiceguard/ ./voiceguard/
COPY models/aasist/ ./models/aasist/
COPY data/demo/ ./data/demo/
COPY data/partials/ ./data/partials/
COPY reports/voiceguard/ ./reports/voiceguard/
COPY --from=frontend /build/dist ./frontend/dist
ENV VOICEGUARD_DATA=/data
EXPOSE 8765
CMD ["python","-m","uvicorn","voiceguard.server:app","--host","0.0.0.0","--port","8765","--ws-max-size","32012","--ws-max-queue","2"]
