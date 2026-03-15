from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Prediction:
    """
    Output contract from infer.py.

    service          : which service this prediction is about
    metric           : which metric the prediction is based on
    score            : failure probability between 0.0 and 1.0
    horizon_seconds  : how far ahead the prediction looks
    metadata         : any extra labels the model wants to surface in Grafana
    """
    service: str
    metric: str
    score: float
    horizon_seconds: int
    metadata: Dict[str, str] = field(default_factory=dict)