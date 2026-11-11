"""Train an exam-score regressor and save preprocessing + weights in one file.

Training:
    python exam_score_model.py --csv cleaned_data.csv --output exam_score.safetensors

Inference (no training CSV or separate scaler is needed):
    model = ExamScoreRegressor.load("exam_score.safetensors")
    scores = model.predict([student_dictionary])

Inputs use the names, units, and numeric/Boolean encodings in cleaned_data.csv.
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split


TARGET = "exam_score"


class ExamScoreRegressor(nn.Module):
    """Accept original numeric features and encode course without imposing order."""

    def __init__(self, numeric_columns, course_categories, hidden_sizes=(64, 32)):
        super().__init__()
        self.numeric_columns = list(numeric_columns)
        self.course_categories = list(course_categories)
        self.hidden_sizes = list(hidden_sizes)

        n_numeric = len(self.numeric_columns)
        self.register_buffer("feature_mean", torch.zeros(n_numeric))
        self.register_buffer("feature_scale", torch.ones(n_numeric))

        # One-hot course features are appended after the standardized numbers.
        width = n_numeric + len(self.course_categories)
        layers = []
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(width, hidden), nn.ReLU()])
            width = hidden
        layers.append(nn.Linear(width, 1))  # Regression: one unrestricted score.
        self.network = nn.Sequential(*layers)

    @torch.no_grad()
    def fit_normalization(self, training_features):
        """Call ONCE before training, with training rows only.

        This is feature-wise standardization. It also standardizes the binary
        columns, which is valid; their original values remain unchanged in X.
        """
        mean = training_features.mean(dim=0)
        std = training_features.std(dim=0, correction=0)
        scale = torch.where(std == 0, torch.ones_like(std), std)
        self.feature_mean.copy_(mean)
        self.feature_scale.copy_(scale)

    def forward(self, raw_numeric, course_ids):
        # These buffers stay fixed during training, validation, and inference.
        standardized = (raw_numeric - self.feature_mean) / self.feature_scale
        course_one_hot = F.one_hot(
            course_ids, num_classes=len(self.course_categories)
        ).to(dtype=raw_numeric.dtype)
        features = torch.cat([standardized, course_one_hot], dim=1)
        return self.network(features).squeeze(-1)  # (batch,), matching y.

    def encode(self, records):
        """Convert a DataFrame, list of dictionaries, or one dictionary to tensors.

        This selects columns by saved name, so input dictionary/column order
        cannot silently change the meaning of a feature. Extra columns such
        as exam_score are ignored; exam_score never enters the network.
        """
        if isinstance(records, dict):
            records = [records]
        frame = pd.DataFrame(records)
        required = ["course", *self.numeric_columns]
        missing = [name for name in required if name not in frame.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        if frame[required].isna().any().any():
            raise ValueError("Input features contain missing values.")

        numeric = frame[self.numeric_columns].apply(pd.to_numeric, errors="raise")
        raw = torch.tensor(numeric.to_numpy(dtype="float32"))
        if not torch.isfinite(raw).all():
            raise ValueError("Numeric features must be finite.")

        category_map = {name: i for i, name in enumerate(self.course_categories)}
        ids = frame["course"].map(category_map)
        if ids.isna().any():
            unknown = frame.loc[ids.isna(), "course"].unique().tolist()
            raise ValueError(
                f"Unknown course values {unknown}; expected {self.course_categories}"
            )
        return raw, torch.tensor(ids.to_numpy(dtype="int64"))

    @torch.inference_mode()
    def predict(self, records):
        """Return exam-score predictions in the ORIGINAL score units on CPU."""
        self.eval()
        numeric, course_ids = self.encode(records)
        device = self.feature_mean.device
        return self(numeric.to(device), course_ids.to(device)).cpu()

    def save(self, filename):
        """One safetensors file contains weights, buffers, and input schema."""
        config = {
            "format_version": 1,
            "target": TARGET,
            "numeric_columns": self.numeric_columns,
            "course_categories": self.course_categories,
            "hidden_sizes": self.hidden_sizes,
        }
        tensors = {
            name: value.detach().cpu().contiguous()
            for name, value in self.state_dict().items()
        }
        save_file(tensors, str(filename), metadata={"config": json.dumps(config)})

    @classmethod
    def load(cls, filename):
        with safe_open(str(filename), framework="pt", device="cpu") as handle:
            config = json.loads(handle.metadata()["config"])
        if config["format_version"] != 1:
            raise ValueError("Unsupported model format version")
        model = cls(
            config["numeric_columns"],
            config["course_categories"],
            config["hidden_sizes"],
        )
        model.load_state_dict(load_file(str(filename), device="cpu"))
        model.eval()
        return model


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    squared_error, absolute_error, count = 0.0, 0.0, 0
    for numeric, course_ids, target in loader:
        error = model(numeric, course_ids) - target
        squared_error += error.square().sum().item()
        absolute_error += error.abs().sum().item()
        count += len(target)
    return {"mse": squared_error / count, "rmse": math.sqrt(squared_error / count),
            "mae": absolute_error / count}


def train(csv_path, output_path, epochs=150, patience=20, seed=42):
    torch.manual_seed(seed)
    frame = pd.read_csv(csv_path)
    if TARGET not in frame or "course" not in frame:
        raise ValueError("CSV must include course and exam_score columns.")
    if len(frame) < 20:
        raise ValueError("Provide at least 20 rows for this three-way split.")
    if frame.isna().any().any():
        raise ValueError("This example requires a cleaned CSV with no missing values.")

    # Split row indices FIRST. No target or feature statistics are fitted yet.
    train_rows, val_rows, test_rows = random_split(
        range(len(frame)), [0.70, 0.15, 0.15],
        generator=torch.Generator().manual_seed(seed),
    )
    numeric_columns = [c for c in frame.columns if c not in (TARGET, "course")]
    course_categories = sorted(frame.iloc[train_rows.indices]["course"].unique())
    model = ExamScoreRegressor(numeric_columns, course_categories)

    X, course_ids = model.encode(frame)
    y = torch.tensor(frame[TARGET].to_numpy(dtype="float32"))
    if not torch.isfinite(y).all():
        raise ValueError("Targets must be finite.")
    model.fit_normalization(X[train_rows.indices])

    # Starting near the training mean speeds convergence. y is never scaled.
    training_target_mean = y[train_rows.indices].mean()
    with torch.no_grad():
        model.network[-1].bias.copy_(training_target_mean.reshape(1))

    dataset = TensorDataset(X, course_ids, y)
    train_loader = DataLoader(
        Subset(dataset, train_rows.indices), batch_size=256, shuffle=True,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    val_loader = DataLoader(Subset(dataset, val_rows.indices), batch_size=512)
    test_loader = DataLoader(Subset(dataset, test_rows.indices), batch_size=512)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_loss, best_state, best_epoch, stale_epochs = math.inf, None, 0, 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for numeric, ids, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(numeric, ids)
            loss = loss_fn(prediction, target)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(target)

        validation = evaluate(model, val_loader)
        train_rmse = math.sqrt(train_loss_sum / len(train_rows))
        history.append({"epoch": epoch, "train_rmse": train_rmse,
                        "validation_rmse": validation["rmse"]})
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | train RMSE {train_rmse:.3f} | "
                  f"validation RMSE {validation['rmse']:.3f}", flush=True)

        if validation["mse"] < best_loss:
            best_loss, best_epoch, stale_epochs = validation["mse"], epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping at epoch {epoch}.", flush=True)
                break

    # Restore the validation-selected checkpoint BEFORE looking at test metrics.
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader)
    y_test = y[test_rows.indices]
    test_variance = ((y_test - y_test.mean()) ** 2).mean().item()
    test_metrics["r2"] = 1 - test_metrics["mse"] / test_variance
    baseline_mse = ((y_test - training_target_mean) ** 2).mean().item()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)

    # Verify that the single saved artifact restores identical predictions.
    loaded = ExamScoreRegressor.load(output_path)
    examples = frame.iloc[test_rows.indices[:32]].drop(columns=TARGET)
    before = model.predict(examples)
    after = loaded.predict(examples)
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    torch.testing.assert_close(
        loaded.predict(examples[examples.columns[::-1]]), before, rtol=0, atol=0
    )

    report = {
        "source_csv": str(Path(csv_path).resolve()), "seed": seed,
        "rows": len(frame),
        "split_counts": {"train": len(train_rows), "validation": len(val_rows),
                         "test": len(test_rows)},
        "raw_feature_count": len(numeric_columns) + 1,
        "network_input_count": len(numeric_columns) + len(course_categories),
        "numeric_columns": numeric_columns, "course_categories": course_categories,
        "hidden_sizes": model.hidden_sizes,
        "best_epoch": best_epoch, "epochs_run": epoch,
        "best_validation_rmse": math.sqrt(best_loss),
        "test": test_metrics, "training_mean_baseline_test_rmse": math.sqrt(baseline_mse),
        "save_load_predictions_identical": True,
        "sample_test_row_index": test_rows.indices[0],
        "sample_prediction": after[0].item(),
        "sample_actual": float(frame.iloc[test_rows.indices[0]][TARGET]),
        "torch_version": torch.__version__, "pandas_version": pd.__version__,
        "history": history,
    }
    report_path = output_path.with_suffix(".metrics.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))
    print(f"Saved model: {output_path}\nSaved metrics: {report_path}")
    return model, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("exam_score.safetensors"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.threads) < 1:
        parser.error("epochs, patience, and threads must be positive")
    torch.set_num_threads(args.threads)
    train(args.csv, args.output, args.epochs, args.patience, args.seed)
