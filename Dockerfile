FROM python:3.12-slim
WORKDIR /main
USER root
COPY . .

RUN apt-get update && apt-get install -y ca-certificates dos2unix
RUN pip3 install --no-cache-dir -r requirements.txt
RUN dos2unix /main/*

EXPOSE 2016
WORKDIR /main/scripts
CMD ["bash", "serve.sh"]