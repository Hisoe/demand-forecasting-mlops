# Use an official lightweight Python runtime
FROM python:3.11-slim

# Set system environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set the working directory inside the container
WORKDIR /app

# Copy dependencies first to leverage Docker's caching mechanism
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API source code into the container
COPY api/ ./api

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the Uvicorn server when the container boots
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
