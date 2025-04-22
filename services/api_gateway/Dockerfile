

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app
WORKDIR $APP_HOME

# Install dependencies for this service
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy core module first (adjust path if needed)
COPY ../../core /app/core

# Copy this service's application code
COPY ./app /app/app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]