# Use an official Python image
FROM python:3.9-slim

# Install system dependencies (poppler)
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV STREAMLIT_DISABLE_TELEMETRY="1"
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose the port Streamlit runs on
EXPOSE 7860

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
