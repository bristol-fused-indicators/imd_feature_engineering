from enum import StrEnum
from dataclasses import dataclass


class ReductionMethod(StrEnum):
    NONE = "none"
    PCA = "pca"
    NMF = "nmf"
    FA = "factor_analysis"
    ICA = "ica"


@dataclass
class GroupConfig:
    columns: list[str]
    scale: bool = False
    reduction_method: ReductionMethod = ReductionMethod.NONE
    n_components: int | None = None


@dataclass
class FeatureSetConfig:
    name: str
    groups: dict[str, GroupConfig]
    description: str = ""
