# Dockerfile - this is a comment. Delete me if you want.
FROM python:3.7.4

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

# define the port number the container should expose
EXPOSE 5000

CMD ["python", "app.py"]

