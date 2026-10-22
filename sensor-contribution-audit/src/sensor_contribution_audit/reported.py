"""Export attributed article transcriptions; no statistical estimation."""
import csv
import json
from importlib.resources import files


def reported_values():
    base = files("sensor_contribution_audit").joinpath("data/reported")
    source_map = json.loads(base.joinpath("source_map.json").read_text(encoding="utf-8"))
    tables = {}
    for article in source_map["source_articles"]:
        for name in article["files"]:
            with base.joinpath(name).open("r", encoding="utf-8", newline="") as stream:
                tables[name] = list(csv.DictReader(stream))
    return {
        "reported_values_only": True,
        "new_statistics": False,
        "source_map": source_map,
        "tables": tables,
    }
