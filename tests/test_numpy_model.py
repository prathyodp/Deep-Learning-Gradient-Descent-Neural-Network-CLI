import numpy as np
from src.models import NumpyNetwork

def test_linear_training_reduces_loss():
    rng=np.random.default_rng(7)
    X=rng.normal(size=(100,2)).astype(np.float32)
    y=(2*X[:,[0]]-X[:,[1]]+1).astype(np.float32)
    model=NumpyNetwork(2, model="linear", seed=7)
    before=np.mean((model.predict(X)-y)**2)
    model.fit(X,y,epochs=300,lr=0.05)
    after=np.mean((model.predict(X)-y)**2)
    assert after < before
