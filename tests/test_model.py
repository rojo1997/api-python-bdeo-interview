"""Test image classfier"""
import sys
from decouple import config
from fastapi.testclient import TestClient
import numpy as np

sys.path.append("./app")

from app.main import app
from app.core import Model


def test_model_prediction():
    """Test 1: scene classification.

    Should return a dict like this:
    {'class': 'predicted_class'}
    """
    app.state.model = Model(
        filename=config("MODEL_FILENAME", cast=str, default="weights/doubleit_model.pt")
    )
    client = TestClient(app)
    array = np.random.randint(0, 10, (4))
    response = client.post(url="/prediction", json={"values": array.tolist()})
    assert response.ok, str(response.json())
    assert response.json() == {"values": (array * 2).tolist()}
