import pytest
import numpy as np
from flexdamage.models.symbolic import SymbolicModel

def test_symbolic_model_parsing():
    # Test valid formula
    model = SymbolicModel("alpha * x + beta * x**2", {"x": "temp"})
    assert "alpha" in model.get_parameter_names()
    assert "beta" in model.get_parameter_names()
    assert len(model.data_symbols) == 1
    
    # Test built-in function collision handling (beta)
    model = SymbolicModel("beta * x", {"x": "temp"})
    assert "beta" in model.get_parameter_names()

def test_symbolic_model_matrix(dummy_data):
    model = SymbolicModel("alpha * x + beta * x**2", {"x": "temperature_anomaly"})
    X, params = model.prepare_design_matrix(dummy_data)
    
    assert X.shape == (100, 2)
    assert params == ["alpha", "beta"]
    
    # Check values: column 0 should be x, column 1 should be x^2
    np.testing.assert_allclose(X[:, 0], dummy_data["temperature_anomaly"])
    np.testing.assert_allclose(X[:, 1], dummy_data["temperature_anomaly"]**2)

def test_missing_column(dummy_data):
    model = SymbolicModel("alpha * x", {"x": "missing_col"})
    with pytest.raises(ValueError):
        model.prepare_design_matrix(dummy_data)
