"""FeatureHub Registry for entities and feature views."""

from __future__ import annotations

import json
import threading
from typing import Any, Dict, List, Optional
from featurehub.metadata import DataType, Entity, Feature, FeatureView


class Registry:
    """Thread-safe registry for FeatureHub entities and feature views."""

    def __init__(self):
        self._entities: Dict[str, Entity] = {}
        self._feature_views: Dict[str, FeatureView] = {}
        self._lock = threading.RLock()

    def register_entity(self, entity: Entity) -> Entity:
        with self._lock:
            self._entities[entity.name] = entity
            return entity

    def get_entity(self, name: str) -> Optional[Entity]:
        with self._lock:
            return self._entities.get(name)

    def list_entities(self) -> List[Entity]:
        with self._lock:
            return list(self._entities.values())

    def register_feature_view(self, feature_view: FeatureView) -> FeatureView:
        with self._lock:
            # Validate that entity exists
            if feature_view.entity_name not in self._entities:
                raise ValueError(
                    f"Entity '{feature_view.entity_name}' not registered in registry."
                )
            self._feature_views[feature_view.name] = feature_view
            return feature_view

    def get_feature_view(self, name: str) -> Optional[FeatureView]:
        with self._lock:
            return self._feature_views.get(name)

    def list_feature_views(self) -> List[FeatureView]:
        with self._lock:
            return list(self._feature_views.values())

    def to_dict(self) -> Dict[str, Any]:
        """Serializes registry definitions to a JSON-compatible dict."""
        with self._lock:
            return {
                "entities": [
                    {
                        "name": e.name,
                        "join_keys": e.join_keys,
                        "description": e.description,
                    }
                    for e in self._entities.values()
                ],
                "feature_views": [
                    {
                        "name": fv.name,
                        "entity_name": fv.entity_name,
                        "description": fv.description,
                        "ttl_seconds": fv.ttl_seconds,
                        "online": fv.online,
                        "offline": fv.offline,
                        "features": [
                            {
                                "name": f.name,
                                "dtype": f.dtype.value,
                                "description": f.description,
                                "default_value": f.default_value,
                                "min_value": f.min_value,
                                "max_value": f.max_value,
                            }
                            for f in fv.features
                        ],
                    }
                    for fv in self._feature_views.values()
                ],
            }
