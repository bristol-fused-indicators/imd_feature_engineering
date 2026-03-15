import sys
from pathlib import Path
from project_paths import paths
import json
from imd_features.config import FeatureSetConfig
from imd_features.process import create_feature_set


INPUT_PATH = paths.input_file
OUTPUT_DIR = paths.output


def find_config_files(output_dir: Path) -> list[Path]:
    configs = sorted(output_dir.glob("*_config.json"))
    return configs


def recreate_from_manifest(config_path: Path) -> None:
    manifest = json.loads(config_path.read_text())
    config = FeatureSetConfig.model_validate(manifest)
    df, metadata = create_feature_set(INPUT_PATH, config)
    print(f"  {config.output_name}: {df.shape[0]} rows, {df.shape[1]} columns")


if __name__ == "__main__":
    config_files = find_config_files(OUTPUT_DIR)

    if not config_files:
        print("No config files found")
        sys.exit(1)

    print(f"Recreating {len(config_files)} feature sets\n")
    for config_path in config_files:
        recreate_from_manifest(config_path)
