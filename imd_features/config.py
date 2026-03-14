from enum import StrEnum
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, model_validator


class ReductionMethod(StrEnum):
    NONE = "none"
    PCA = "pca"
    NMF = "nmf"
    FA = "factor_analysis"
    ICA = "ica"


class GroupConfig(BaseModel):
    columns: list[str]
    scale: bool = False
    reduction_method: ReductionMethod = ReductionMethod.NONE
    n_components: int | None = None

    @model_validator(mode="after")
    def validate_n_components_required(self) -> "GroupConfig":
        if self.reduction_method != ReductionMethod.NONE and self.n_components is None:
            raise ValueError(
                f"n_components is required when reduction_method is "
                f"{self.reduction_method.value}"
            )
        return self


class FeatureSetConfig(BaseModel):
    name: str
    groups: dict[str, GroupConfig]
    description: str = ""
