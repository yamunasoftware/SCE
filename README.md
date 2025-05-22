# SCE

Sentiment Classification Engine

## Information

This is a server-based sentiment classification engine, meant to serve as a MLOps (Machine Learning Operations) tool for those that want to classify positive and negative text in digital media, using an API. The first model uses a Logistic Regression model with gradient descent and a Logarithmic Loss function because of its classification use being binary (positive or negative sentiment). The other model uses a SVM (Support Vector Machine) model with an Adam Optimizer and a Hinge Loss function, which is a standard setup for this model on classification tasks.

On the operations side of things, Flask is used to deploy the server backend API, with Docker. There is also an administrative tool built into the classification engine, to access all of the sub-tool scripts like API testing, deploying the server, and training the machine learning model. This tool can be found in the main directory, named ```tool.sh```. All of the sub-tools are found in the folder ```src/scripts```.