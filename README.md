# SCE

Sentiment Classification Engine

## Information

This is a server-based sentiment classification engine, meant to serve as a MLOps (Machine Learning Operations) tool for those that want to classify positive and negative text in digital media, using an API. The model uses a Logistic Regression model with gradient descent and a Logarithmic Loss function because of its classification use being binary (positive or negative sentiment).

On the operations side of things, Flask is used to deploy the server backend API, with Docker. There is also an administrative tool built into the classification engine, to access all of the sub-tool scripts like API testing, deploying the server, and training the machine learning model. This tool can be found in the main directory, named ```tool.sh```. All of the sub-tools are found in the folder ```src/scripts```.

**NOTE: All of the tools are written in Bash scripting, which means that the intended purpose of this tool is to be deployed on a Linux server or a WSL (Windows Subsystem for Linux) system.**

## Docker and Kubernetes

To use Docker, you can follow the following commands to build and deploy the API:

```
docker build -t SCE .
docker run -d -p 5000:5000 SCE
```

To deploy the Docker image on Kubernetes, you can use the provided ```deployment.yaml``` file. To, deploy follow the commands below:

```
kubectl apply -f deployment.yaml
kubectl get deployment
kubectl get pods
kubectl get svc
```

## Dependencies

The python project uses the following python modules:

- os
- sys
- random
- string
- datetime
- unittest
- warnings
- numpy
- sqlite3
- sklearn
- flask
- pandas

However, the only dependencies that need to be installed are ```numpy```, ```sqlite3```, ```sklearn```, ```flask```, and ```pandas```. To install these dependencies you have to run the following commands:

```
pip install numpy
pip install sqlite3
pip install sklearn
pip install flask
pip install pandas
```