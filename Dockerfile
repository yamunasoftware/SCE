FROM python:3.12-slim
WORKDIR /main
USER root
EXPOSE 2016
COPY . .

RUN mkdir -p /main/logs
RUN apt-get update && apt-get install -y ca-certificates dos2unix
RUN pip3 install --no-cache-dir -r requirements.txt
RUN dos2unix /main/*
CMD ["bash", "serve.sh"]