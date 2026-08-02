FROM python:3.10-slim

# Copy the following files and folders
COPY ./app /app
COPY ./requirements.txt /requirements.txt
COPY .env .env
# Copy the AI models
COPY ./*.pt /

# Install system dependencies needed for OpenCV and Scikit-Image
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install python packages
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face Spaces Docker requirement: expose port 7860
EXPOSE 7860

# Run FastAPI backend on port 7860
CMD ["uvicorn", "app.main:app", "--host=0.0.0.0", "--port=7860"]