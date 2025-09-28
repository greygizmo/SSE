#!/usr/bin/env python3
"""Author a placeholder cold-start model artifact for divisions lacking history.

Some downstream services expect a ``model.pkl`` and metadata.json even when a
division has not yet been trained.  This utility writes a scikit-learn dummy
classifier to disk using predictable defaults so that deployments and CI can run
end-to-end while we collect real data.  Example usage::

    python scripts/create_dummy_cold_model.py --division Solidworks \
        --output gosales/models/solidworks_cold_model/model.pkl
"""
import argparse
import json
from pathlib import Path
import joblib

try:
    from sklearn.dummy import DummyClassifier
except Exception:
    DummyClassifier = None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--division", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--strategy", default="prior", choices=["prior", "uniform"])  # constant policy
    args = p.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    if DummyClassifier is None:
        raise SystemExit("scikit-learn not available to create dummy model. Install scikit-learn.")

    # A basic dummy classifier that predicts class probabilities by empirical prior
    # We fit with a trivial dataset with 2 classes to initialize the object.
    import numpy as np
    X = np.zeros((2, 1))
    y = np.array([0, 1])
    clf = DummyClassifier(strategy=args.strategy, random_state=42)
    clf.fit(X, y)
    joblib.dump(clf, out)
    print(f"Wrote dummy cold-start model to {out}")

    # Write minimal metadata alongside if not present
    meta_path = out.parent / "metadata.json"
    if not meta_path.exists():
        meta = {"division": args.division, "model_type": "sklearn_dummy", "feature_names": []}
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Wrote metadata to {meta_path}")


if __name__ == "__main__":
    main()

