# Use the full Python image instead of slim to avoid path/link issues
FROM python:3.11

# Set system environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set the working directory
WORKDIR /app

# Upgrade pip and install core system tools first
RUN pip install --upgrade pip && \
    pip install "setuptools<82.0.0" wheel

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 --retries=20 -r requirements.txt

# Copy the API source code
COPY api/ ./api

# Expose the port 
EXPOSE 8000

# Command to run
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]