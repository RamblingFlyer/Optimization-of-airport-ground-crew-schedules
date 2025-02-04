# Dockerfile
FROM python:3.9
WORKDIR /app
COPY . /app
RUN pip install -r app/requirements.txt
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
