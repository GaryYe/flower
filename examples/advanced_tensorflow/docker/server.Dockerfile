FROM flower:latest

RUN mkdir /app
WORKDIR /app

ADD server.py /app
CMD ["python", "server.py"]
