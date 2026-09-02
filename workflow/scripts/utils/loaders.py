import csv
import json
import logging
import pandas as pd
import yaml

from pathlib import Path

from utils.anonymization_keys import AnonymizationKeys

logger = logging.getLogger()


def load_anon_key_yaml(yaml_path:Path) -> dict:
    if not yaml_path.exists():
        msg = f"Config file {yaml_path} does not exist."
        logger.error(msg)
        raise FileNotFoundError()

    try:
        with yaml_path.open("r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as ye:
        msg = f"Invalid YAML path"
        logger.exception(msg)
        raise ye

    if not config:
        msg = f"YAML file is empty or invalid."
        logger.error(msg)
        raise IOError
    
    return AnonymizationKeys.model_validate(config)


def load_id_map(csv_path:Path) -> dict:
    if not csv_path.exists():
        msg = f"ID Map file {csv_path} does not exist."
        logger.error(msg)
        raise FileNotFoundError()
    
    id_map = {}
    try:
        with csv_path.open("r", encoding='utf-8-sig') as f:
            file_data = csv.reader(f, )
            for row in file_data:
                id_map[row[0]] = row[1]
    except Exception as e:
        msg = f"Error reading {csv_path}"
        logger.exception(msg)
        raise e

    return id_map


def load_crawl_db(json_path:Path) -> pd.DataFrame:
    """Reads the crawl_db.json file and returns a flattened DataFrame with one row per series_id."""
    with json_path.open() as file:
        crawl_db = json.load(file)

    flattened_crawl_db = []
    # crawl db is nested as {series_id: {idx: {dicom_tags}}}, so need to drop that idx in the middle to get a flat structure for the dataframe
    for series_id, series_info in crawl_db.items():
        series_dict = {'series_id': series_id}
        for _idx, info in series_info.items():
            series_dict.update(info)

        flattened_crawl_db.append(series_dict)

    return pd.DataFrame(flattened_crawl_db)