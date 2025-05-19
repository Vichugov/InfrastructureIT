FROM python:3.12.3-slim

ENV PYTHONUNBUFFERED=1
ENV APP_PORT=$PORT

RUN groupadd pygroup && useradd --gid pygroup --create-home pyuser
USER pyuser

WORKDIR ~/app 
COPY . .   
        
RUN pip install --upgrade pip --user
RUN pip install --no-cache-dir -r ./requirements.txt --user

CMD ["python", "app.py"]
EXPOSE $PORT
