# Hugging Face Spaces (Docker SDK) deployment.
# Only the ~92MB derived graph cache (data/graph/) ships in the image - the
# ~900MB of raw Zenodo/GitHub downloads are build-time-only inputs
# (scripts/build_graph.py), not needed at runtime.
FROM python:3.12-slim

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend ./backend
COPY frontend ./frontend
COPY data/graph ./data/graph

WORKDIR /app/backend

# Hugging Face Docker Spaces route external traffic to port 7860 by default.
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
