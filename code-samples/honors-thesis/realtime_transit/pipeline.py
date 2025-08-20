"""
pipeline: Module responsible with data pipeline tasks, i.e. moving data around from component to component
efficiently within the project. This module is designed so that by importing its contents into a script, that
script will have everything it needs to train a model. 

Author: Nick Gable (gable105@umn.edu)
"""

import os
import pandas as pd

from datetime import datetime, timedelta
from typing import Union, List, Generator
from glob import glob

# Imports used in scripts - some may not be used in this module directly but are helpful!
from realtime_transit.util import gtfs_file, GTFS_DIR
from realtime_transit.crossing import calculate_crosses, calculate_crosses_mult, append_schedule
from realtime_transit import graphs
import torch

COLLECTION_START = datetime(2023, 9, 10)
COLLECTION_END = datetime(2023, 9, 30)
COLLECTION_DIR = 'data/collection'


def get_collected_data(data_type: str,
                       filter_routes: Union[List[int], None] = None,
                       start_date: Union[datetime, None] = COLLECTION_START,
                       end_date: Union[datetime, None] = COLLECTION_END,
                       collection_dir: str = COLLECTION_DIR) -> Generator[pd.DataFrame, None, None]:
    """
    Generator that returns data collected from the collections module in one day increments. Use to efficiently access
    full set of collected data without loading it into memory or having to deal with direct file I/O in other parts of the code.

    Parameters:
    - `data_type`: string collected data type, `pred` or `rt-pos`
    - `filter_routes`: optional list of route numbers to filter by
    - `start_date`: optional start date of data: defaults to first day
    - `end_date`: optional end date of data: defaults to last day (inclusive)
    - `collection_dir`: directory where collection data is stored
    """
    current_date = start_date
    while current_date <= end_date:
        path_expr = os.path.join(
            collection_dir, f"{data_type}-{current_date.strftime('%Y-%m-%d')}-*.csv.zip")
        files = glob(path_expr)

        # allow for multiple files on same day, although our collection set doesn't have this
        frames = []
        for file in files:
            frames.append(pd.read_csv(file))

        day_data = pd.concat(frames)
        if filter_routes:
            day_data = day_data[day_data.route_id.isin(filter_routes)]

        current_date += timedelta(days=1)

        yield day_data


def generate_network(filter_routes: Union[List[int], None] = None, gtfs_dir: str = GTFS_DIR) -> graphs.TransitNetwork:
    """
    Generate a TransitNetwork using static GTFS data, optionally filtering by routes in `filter_routes`, optionally specifying
    a different GTFS directory than GTFS_DIR via `gtfs_dir`.
    """
    gtfs_stops = gtfs_file("stops", gtfs_dir)
    gtfs_stop_times = gtfs_file("stop_times", gtfs_dir)
    gtfs_trips = gtfs_file("trips", gtfs_dir)

    if filter_routes:
        gtfs_trips = gtfs_trips[gtfs_trips.route_id.isin(filter_routes)]
        gtfs_stop_times = gtfs_stop_times[gtfs_stop_times.trip_id.isin(
            gtfs_trips.trip_id)]
        gtfs_stops = gtfs_stops[gtfs_stops.stop_id.isin(
            gtfs_stop_times.stop_id)]

    return graphs.create_network(gtfs_stops, gtfs_stop_times, gtfs_trips)


def cuda_init():
    """
    Prepare PyTorch globally for CUDA, checking if CUDA is available and setting it as the default globally if so.
    """
    if torch.cuda.is_available():
        torch.set_default_device('cuda')  # enable cuda
