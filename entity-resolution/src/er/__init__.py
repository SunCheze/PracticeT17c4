"""Entity Resolution: поиск дубликатов клиентских профилей."""
from . import config
from .parsing import flatten, parse_kv_list_to_dict, parse_json_to_dict
from .preprocessing import collapse_to_profiles
from .blocking import build_blocks, candidate_pairs, label_pairs, blocking_recall
from .features import generate_pairwise_features
from .model import train_model, predict_proba, save_model, load_model
from .clustering import cluster_pairs, recommend_action

__version__ = "0.1.0"

__all__ = [
    "flatten", "parse_kv_list_to_dict", "parse_json_to_dict",
    "collapse_to_profiles",
    "build_blocks", "candidate_pairs", "label_pairs", "blocking_recall",
    "generate_pairwise_features",
    "train_model", "predict_proba", "save_model", "load_model",
    "cluster_pairs", "recommend_action",
    "config",
]
