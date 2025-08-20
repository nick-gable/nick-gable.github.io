"""
util: Misc utility functions used throughout the project.

Author: Nick Gable (gable105@umn.edu)
"""

import pandas as pd
import os

GTFS_DIR = 'data/gtfs'


def gtfs_file(file: str, gtfs_dir: str = GTFS_DIR):
    return pd.read_csv(os.path.join(gtfs_dir, f"{file}.txt"))
