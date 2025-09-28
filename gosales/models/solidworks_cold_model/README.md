SolidWorks Cold-Start Model

This folder is a scaffold for an optional cold-start model used to score customers with sparse or no recent transaction behavior for the SolidWorks division. If `model.pkl` is present, the scoring pipeline routes cold rows through this model; otherwise the warm model is used for all rows.

Files:
- model.pkl                Serialized scikit-learn model (joblib)
- metadata.json            Minimal metadata with division and (optional) feature_names
- feature_list.json        Exact feature order expected by the model (recommended)

To generate a quick dummy model for local testing:
1) Create a virtualenv with scikit-learn and joblib installed.
2) Run the helper script:

   python scripts/create_dummy_cold_model.py --division Solidworks --output gosales/models/solidworks_cold_model/model.pkl

This will save a calibrated DummyClassifier that outputs a constant probability. Replace with a properly trained model for production.
