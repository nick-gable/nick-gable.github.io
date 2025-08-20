"""
processing: functions and classes related to taking in raw data feeds and converting them to a neutral format.

Implementation note: Output formats from these functions (such as DataFrame column names) should be treated
as the standard format throughout the project. If a column name change is desired, or a field needs to be added,
it should NOT be done later on in the code - this results in confusing situations that make understanding the code
or data being used a lot more difficult.

Author: Nick Gable (gable105@umn.edu)
"""

from google.transit import gtfs_realtime_pb2
import urllib.request as request
import pandas as pd
from datetime import datetime

TRIP_UPDATES = "https://svc.metrotransit.org/mtgtfs/tripupdates.pb"
VEHICLE_POSITION = "https://svc.metrotransit.org/mtgtfs/vehiclepositions.pb"


def get_realtime_locations() -> pd.DataFrame:
    """
    Returns pandas DataFrame containing vehicle locations as provided by 
    Metro Transit GTFS-realtime feed.

    Output format is a DataFrame with the following labels containing the
    matching field from the GTFS realtime feed:

    | Field       | Data type |
    |-------------|-----------|
    | `trip_id`   | `str`     |
    | `route_id`  | `int`     |
    | `latitude`  | `float64` |
    | `longitude` | `float64` |
    | `bearing`   | `float64` |
    | `speed`     | `float64` |

    Additionally, the DataFrame contains a field `date` of type `datetime` representing
    the time that the feed was polled.
    """
    # read in GTFS-realtime locations feed
    vehicle_positions = gtfs_realtime_pb2.FeedMessage()
    response = request.urlopen(VEHICLE_POSITION)
    vehicle_positions.ParseFromString(response.read())

    # populate with realtime data, then transform into DataFrame
    current_date = datetime.now()

    acc_dict = {
        'date': [],
        'trip_id': [],
        'route_id': [],
        'latitude': [],
        'longitude': [],
        'bearing': [],
        'speed': []
    }

    for entity in vehicle_positions.entity:
        vehicle = entity.vehicle

        acc_dict['trip_id'].append(vehicle.trip.trip_id)
        acc_dict['route_id'].append(int(vehicle.trip.route_id))
        acc_dict['latitude'].append(vehicle.position.latitude)
        acc_dict['longitude'].append(vehicle.position.longitude)
        acc_dict['bearing'].append(vehicle.position.bearing)
        acc_dict['speed'].append(vehicle.position.speed)
        # TODO consider using reported timestamp instead or adding field
        acc_dict['date'].append(current_date)

    return pd.DataFrame.from_dict(acc_dict)


def get_trip_updates() -> pd.DataFrame:
    """
    Returns pandas DataFrame containing predicted departure times at stops across
    the Metro Transit network, as provided by their GTFS-realtime feed. 

    Note that this function only returns actual predictions - stops that currently have no
    data (likely due to a trip that isn't reporting vehicle location) are not reported.

    Output format is a DataFrame with the following labels:

    | Field                 | Data type  | Description
    |-----------------------|------------|-------------------------------------------------------------
    | `date`                | `datetime` | Feed poll date
    | `trip_id`             | `str`      | GTFS `trip_id`
    | `route_id`            | `int`      | GTFS `route_id`
    | `stop_sequence`       | `int`      | GTFS `stop_sequence`
    | `stop_id`             | `int`      | GTFS `stop_id`
    | `predicted_delay`     | `int`      | Predicted schedule deviance for this trip / stop, in seconds
    | `predicted_departure` | `datetime` | Predicted departure time for this trip / stop
    """
    trip_updates = gtfs_realtime_pb2.FeedMessage()
    response = request.urlopen(TRIP_UPDATES)
    trip_updates.ParseFromString(response.read())

    current_date = datetime.now()

    acc_dict = {
        'date': [],
        'trip_id': [],
        'route_id': [],
        'stop_sequence': [],
        'stop_id': [],
        'predicted_delay': [],
        'predicted_departure': []
    }

    # iterate over each update, containing multiple stop_time_update items
    for entity in trip_updates.entity:
        trip_id = entity.trip_update.trip.trip_id
        route_id = int(entity.trip_update.trip.route_id)

        for stop_time_update in entity.trip_update.stop_time_update:
            if stop_time_update.departure.time == 0:
                # no data here, skip
                continue

            acc_dict['date'].append(current_date)
            acc_dict['trip_id'].append(trip_id)
            acc_dict['route_id'].append(route_id)
            acc_dict['stop_sequence'].append(stop_time_update.stop_sequence)
            acc_dict['stop_id'].append(int(stop_time_update.stop_id))
            acc_dict['predicted_delay'].append(
                stop_time_update.departure.delay)
            acc_dict['predicted_departure'].append(
                datetime.fromtimestamp(stop_time_update.departure.time)
            )

    return pd.DataFrame.from_dict(acc_dict)
