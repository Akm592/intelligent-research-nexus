# services/ui_service/Dockerfile

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV GRADIO_SERVER_NAME=0.0.0.0

ENV APP_HOME /app
WORKDIR $APP_HOME

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ../../core /app/core
COPY ./app /app/app

EXPOSE 7860 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]