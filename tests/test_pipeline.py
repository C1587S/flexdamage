import pytest
from flexdamage.config.loader import load_config
from flexdamage.core.pipeline import EstimationPipeline

def test_config_loading(config_file):
    path, _ = config_file
    cfg = load_config(path)
    assert cfg.run.name == "test_run"
    assert cfg.estimation.functional_form.type == "explicit"

def test_full_pipeline(config_file, dummy_data):
    path, data_dir = config_file
    dummy_data.to_csv(data_dir / "test.csv", index=False)
    
    cfg = load_config(path)
    pipeline = EstimationPipeline(cfg)
    
    # Should run without error
    pipeline.run()
    
    # Check outputs
    out_dir = cfg.run.output_dir
    # Path object from string needed if using pathlib logic
    # But cfg.run.output_dir is a string in the model, let's cast
    from pathlib import Path
    out = Path(out_dir)
    
    assert (out / "global_results.json").exists()
    assert (out / "regional_results.csv").exists()
