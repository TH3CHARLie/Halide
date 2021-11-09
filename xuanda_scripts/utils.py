import json
import os
import subprocess
from typing import List

REQUIRED_KEYS = ["HALIDE_ROOT_PATH", "AUTOSCHED_BIN", "HALIDE_DISTRIB_PATH", "HL_MACHINE_PARAMS"]


def set_up_environment(config_file: str):
    with open(config_file, "r") as f:
        data = json.load(f)
        for key in REQUIRED_KEYS:
            assert key in data, f"{key} is not presented in config file"
            os.environ[key] = data[key]

    host_target = get_host_target()
    print(f"setting HL_TARGET to {host_target}")
    os.environ["HL_TARGET"] = host_target


def get_host_target(excludes: List[str] = []) -> str:
    assert "AUTOSCHED_BIN" in os.environ, "AUTOSCHED_BIN is not set"
    excludes_str = " ".join(excludes)
    output = subprocess.check_output(f"{os.environ['AUTOSCHED_BIN']}/get_host_target {excludes_str}", shell=True)
    return output.decode('utf-8')