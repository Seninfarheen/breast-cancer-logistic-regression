from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_model(X_train, y_train, reg=False):
  
    """
    Train a logistic regression model with or without L2 regularization.
    
    Args:
        X_train (array): Feature matrix
        y_train (array): Labels
        reg (bool): If True, use L2 regularization

    Returns:
        model: Trained sklearn model
    """
  
    model = LogisticRegression(
        penalty='l2' if reg else 'none',
        solver='saga',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
  
    """
    Evaluate model on test data and return accuracy.
    """
  
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)
