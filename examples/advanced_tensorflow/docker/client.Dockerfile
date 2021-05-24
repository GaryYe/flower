FROM flower:latest

RUN mkdir /app
WORKDIR /app

ADD client.py /app
# TODO: Argument ..
CMD ["python", "client.py", ]
