FROM ubuntu:latest
WORKDIR /main
USER root
COPY . .

RUN apt-get update
RUN apt-get install -y python3-pip libpq-dev libnuma-dev libsasl2-modules-gssapi-mit ca-certificates openssl
RUN pip3 install --no-cache-dir -r requirements.txt
CMD ["bash", "deploy.sh"]