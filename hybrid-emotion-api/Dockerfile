# Use a lightweight Python image
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies required for librosa and soundfile
RUN apt-get update && apt-get install -y libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them, plus the huggingface_hub library
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt huggingface_hub

# THE MAGIC TRICK: Download your massive weights directly into the container!
RUN huggingface-cli download faiz06/hybrid-emotion-weights --local-dir .

# Copy the rest of your app files into the container
COPY . .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]