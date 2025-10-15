import mlflow.sklearn

def predict(sample):
    model = mlflow.sklearn.load_model("models:/iris-classifier/1")
    return model.predict([sample])[0]
