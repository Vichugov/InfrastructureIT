FROM python:3.12.3-slim

ENV PYTHONUNBUFFERED=1
WORKDIR ./
    
RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "test.py"]
