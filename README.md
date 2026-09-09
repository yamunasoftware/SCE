# SCE

Sentiment Classification Engine

## Information

This is a server-based sentiment classification engine, meant to serve as a REST API for those that want to classify positive and negative text in digital media, with MLOps built-in. The first model uses a Logistic Regression model with gradient descent and a Logarithmic Loss function because of its classification use being binary (positive or negative sentiment). The other model uses a SVM (Support Vector Machine) model with the Hinge Loss function, which is a standard setup for this model on classification tasks.

On the operations side of things, Flask is used to deploy the server backend API, with Docker. The system logs data to stderr and stdout so that container logs can be captured by logging and monitoring tools like Datadog and Prometheus. The full system is meant to be containerized and run on port 8000, on your compute instance (Docker, Kubernetes, ECS, EKS, Azure CA, etc).