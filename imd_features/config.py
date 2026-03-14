from enum import StrEnum
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import hashlib

from pydantic import BaseModel, ConfigDict, model_validator


class ReductionMethod(StrEnum):
    NONE = "none"
    PCA = "pca"
    NMF = "nmf"
    FA = "factor_analysis"
    ICA = "ica"


class GroupConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    columns: list[str]
    scale: bool = False
    reduction_method: ReductionMethod = ReductionMethod.NONE
    n_components: int | None = None

    @model_validator(mode="after")
    def validate_n_components(self) -> "GroupConfig":
        if self.reduction_method != ReductionMethod.NONE and self.n_components is None:
            raise ValueError(
                f"n_components is required when reduction_method is "
                f"{self.reduction_method.value}"
            )
        elif (
            self.reduction_method != ReductionMethod.NONE
            and self.n_components is not None
            and self.n_components <= 0
        ):
            raise ValueError("n_components must be a positive integer")
        elif (
            self.reduction_method == ReductionMethod.NONE
            and self.n_components is not None
        ):
            raise ValueError(
                "do not set n_components without setting a reduction method"
            )

        return self

    @model_validator(mode="after")
    def validate_columns(self) -> "GroupConfig":
        if len(self.columns) < 1:
            raise ValueError("must provide at least one column to a GroupConfig")

        return self


class FeatureSetConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    groups: dict[str, GroupConfig]
    description: str | None

    @model_validator(mode="after")
    def validate_no_group_overlap(self):
        seen: dict[str, str] = {}
        for group_name, group_config in self.groups.items():
            for col in group_config.columns:
                if col in seen:
                    raise ValueError(
                        f"Column '{col}' appears in both group '{seen[col]}' "
                        f"and group '{group_name}'. Each column may only belong "
                        f"to one group."
                    )
                seen[col] = group_name

        return self

    @property
    def config_hash(self) -> str:
        config_bytes = json.dumps(self.model_dump(), sort_keys=True).encode()
        return hashlib.sha256(config_bytes).hexdigest()[:8]

    def create_manifest_dict(self, input_data_hash: str) -> dict:
        return {
            **self.model_dump(),
            "config_hash": self.config_hash,
            "input_data_hash": input_data_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    @property
    def output_name(self) -> str:
        return f"{self.name}_{self.config_hash}"
