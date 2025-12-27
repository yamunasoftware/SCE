# SCE

Sentiment Classification Engine

## Information

This is a server-based sentiment classification engine, meant to serve as a REST API for those that want to classify positive and negative text in digital media, with MLOps built-in. The first model uses a Logistic Regression model with gradient descent and a Logarithmic Loss function because of its classification use being binary (positive or negative sentiment). The other model uses a SVM (Support Vector Machine) model with an Adam Optimizer and a Hinge Loss function, which is a standard setup for this model on classification tasks.

On the operations side of things, Flask is used to deploy the server backend API, with Docker. All of the sub-tools are found in the folder ```src/scripts```. These scripts allow you to manage the ML models (by retraining them), restarting the API service, and clearing the API logs. The full system is meant to be containerized and run on port 2016, on your compute instance. The container is compatible with every major container management platform; therefore, the deployment process should be very straightforward.