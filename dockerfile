FROM alpine:latest
RUN apk update
RUN apk add py-pip
RUN apk add --no-cache python3-dev 
RUN pip install --upgrade pip
WORKDIR /main
COPY . /main
RUN pip --no-cache-dir install -r requirements.txt
CMD ["bash", "deploy.sh"]