"""Metadata definitions for FeatureHub entities, features, and feature views."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union


class DataType(str, Enum):
    INT64 = "INT64"
    FLOAT64 = "FLOAT64"
    STRING = "STRING"
    BOOLEAN = "BOOLEAN"
    VECTOR = "VECTOR"
    JSON = "JSON"


@dataclass
class Feature:
    """Definition of an individual feature attribute."""
    name: str
    dtype: DataType
    description: str = ""
    default_value: Any = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def validate(self, value: Any) -> bool:
        """Validates feature value type and bounds."""
        if value is None:
            return True
        if self.dtype == DataType.INT64 and not isinstance(value, (int,)):
            return False
        if self.dtype == DataType.FLOAT64 and not isinstance(value, (int, float)):
            return False
        if self.dtype == DataType.STRING and not isinstance(value, str):
            return False
        if self.dtype == DataType.BOOLEAN and not isinstance(value, bool):
            return False
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        return True


@dataclass
class Entity:
    """An entity represents the primary subject for features (e.g., user, merchant, device)."""
    name: str
    join_keys: List[str]
    description: str = ""


@dataclass
class FeatureView:
    """A logical grouping of features associated with an entity, with unified schema and TTL."""
    name: str
    entity_name: str
    features: List[Feature]
    description: str = ""
    ttl_seconds: Optional[int] = None  # None means infinite / latest valid
    transformation_name: Optional[str] = None
    online: bool = True
    offline: bool = True

    def get_feature(self, name: str) -> Optional[Feature]:
        for f in self.features:
            if f.name == name:
                return f
        return None

    def feature_names(self) -> List[str]:
        return [f.name for f in self.features]


@dataclass
class FeatureRecord:
    """A single observation or update containing entity key, feature values, and timestamps."""
    entity_key: str
    feature_view_name: str
    values: Dict[str, Any]
    event_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
