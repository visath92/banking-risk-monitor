"""Evaluates detection quality against the ground-truth labels written by the
data generator. This is the ONLY module that reads data/ground_truth/labels.csv
-- no detection/feature code may import from here or read that path, so
evaluation numbers stay honest.
"""
import pandas as pd

from src import config

_labels_cache = None


def _load_labels() -> pd.DataFrame:
    global _labels_cache
    if _labels_cache is None:
        _labels_cache = pd.read_csv(config.GROUND_TRUTH_PATH)
    return _labels_cache


class EvalResult:
    def __init__(self, dataset: str, tp: int, fp: int, fn: int, tn: int, by_type: pd.DataFrame):
        self.dataset = dataset
        self.tp, self.fp, self.fn, self.tn = tp, fp, fn, tn
        self.by_type = by_type

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "true_positives": self.tp,
            "false_positives": self.fp,
            "false_negatives": self.fn,
            "true_negatives": self.tn,
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
            "f1": round(self.f1, 3),
        }


def evaluate(dataset_name: str, predictions_df: pd.DataFrame, id_col: str, predicted_flag_col: str) -> EvalResult:
    labels = _load_labels()
    dataset_labels = labels[labels["dataset"] == dataset_name]
    anomalous_ids = set(dataset_labels.loc[dataset_labels["is_anomaly"], "record_id"])

    df = predictions_df[[id_col, predicted_flag_col]].copy()
    df["actual_anomaly"] = df[id_col].isin(anomalous_ids)
    df["predicted"] = df[predicted_flag_col].astype(bool)

    tp = int((df["actual_anomaly"] & df["predicted"]).sum())
    fp = int((~df["actual_anomaly"] & df["predicted"]).sum())
    fn = int((df["actual_anomaly"] & ~df["predicted"]).sum())
    tn = int((~df["actual_anomaly"] & ~df["predicted"]).sum())

    type_map = dataset_labels.groupby("record_id")["anomaly_type"].agg(lambda types: "+".join(sorted(set(types))))
    df["anomaly_type"] = df[id_col].map(type_map)

    by_type_rows = []
    for anomaly_type, group in df[df["actual_anomaly"]].groupby("anomaly_type"):
        recall = group["predicted"].mean()
        by_type_rows.append({"anomaly_type": anomaly_type, "count": len(group), "recall": round(recall, 3)})
    by_type = pd.DataFrame(by_type_rows, columns=["anomaly_type", "count", "recall"])

    return EvalResult(dataset_name, tp, fp, fn, tn, by_type)
