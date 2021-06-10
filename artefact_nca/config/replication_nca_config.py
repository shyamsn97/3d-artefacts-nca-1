from pydantic.dataclasses import dataclass
from dataclasses import field
from omegaconf import MISSING  # Do not confuse with dataclass.MISSING
from typing import Any, Dict, List, Optional

from artefact_nca.config.base import make_trainer_defaults
from artefact_nca.config.voxel_nca_config import VoxelCATrainerConfig, VoxelCADatasetConfig
from artefact_nca.config.config_utils import add_configs


@dataclass
class ReplicationNCADatasetConfig(VoxelCADatasetConfig):
    cluster_seed: bool = True


trainer_defaults = [{"model_config": "voxel"}, {"dataset_config": "replication"}]

@dataclass
class ReplicationNCATrainerConfig(VoxelCATrainerConfig):
    _target_: str = "artefact_nca.trainer.replication_nca_trainer.ReplicationNCATrainer"
    defaults: List[Any] = field(
        default_factory=lambda: make_trainer_defaults(overrides=trainer_defaults)
    )
    n_duplications: int = 5
    steps_per_duplication: int = 8
    norm_grad: bool = False
    use_sample_pool: bool = False

config_defaults = [{"trainer": "replication"}]

@dataclass
class ReplicationNCAConfig:

    defaults: List[Any] = field(default_factory=lambda: config_defaults)
    trainer: Any = MISSING


config_dicts: List[Dict[str, Any]] = [
    dict(group="trainer/dataset_config", name="replication", node=ReplicationNCADatasetConfig),
    dict(group="trainer", name="replication", node=ReplicationNCATrainerConfig),
    dict(name="replication", node=ReplicationNCAConfig),
]

add_configs(config_dicts)
