# services/kg_service/Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# Set application home directory
ENV APP_HOME /app

# Create and set the working directory
RUN mkdir -p $APP_HOME
WORKDIR $APP_HOME

# Install system dependencies if any are needed by your Python packages
# Example: RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# --- Dependency Installation ---
# Copy only the requirements file first to leverage Docker cache
COPY ./services/kg_service/requirements.txt .

# Install Python dependencies specified in requirements.txt
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Application Code ---
# Copy the shared core library files needed by the service
# Adjust path based on where you run `docker build` (repo root assumed here)
COPY ./core /app/core

# Copy the specific service's application code
COPY ./services/kg_service/app /app/app

# --- Networking ---
# Expose the port the app will run on (must match the CMD port)
EXPOSE 8005

# --- Runtime Command ---
# Use Uvicorn for running the FastAPI application.
# Set --host 0.0.0.0 to allow connections from outside the container.
# Add --reload flag only for development images if needed.
# For production, consider using Gunicorn with Uvicorn workers:
# CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:8005"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005"]