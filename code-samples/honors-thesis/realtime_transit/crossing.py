"""
crossing: Post-process realtime GTFS data, converting it to stop crossing data.

Author: Nick Gable (gable105@umn.edu)
"""

from realtime_transit.util import gtfs_file
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm
from geopy.distance import geodesic as GD
from datetime import datetime, timedelta
from pathos.pools import ProcessPool

GTFS_DIR = 'data/gtfs'

VISIT_RADIUS = 0.1  # distance in miles for when we consider vehicle to be "at the stop"

# maximum tolerable avg. speed difference between the schedule and our reporting
# data points that suggest a differential higher than this will be skipped
MAX_SPEED_DIFFERENTIAL = 2.0

# maximum missed stops before the algorithm will discard the remainder of the trip unless vehicles start
# showing up earlier in the route
MAX_MISSED_STOPS = 5

# in minutes, earliest difference between when a trip is recorded as starting, and when it actually starts (ex. -2 is two minutes early max)
EARLIEST_START = -2


def calculate_crosses(data: pd.DataFrame,
                      fill_last: int = 0,
                      show_progress: bool = True) -> pd.DataFrame:
    """Using the live data fed in from parameter `data`, calculate stop crossing information, and return it in
    a new DataFrame.

    Note: This function will work best when data fed into it includes the start of routes in question - otherwise, the 
    algorithm will have to guess at or not include stop crossing data from earlier in the route.

    Input parameters:
    - `data`: pd.DataFrame containing realtime position data (from `processing.get_realtime_locations()`)
    - `fill_last`: number of stops that should be auto-filled with the last recorded time of the vehicle - use this if you
      know all trips in `data` have all available data (that all trips have ended). Auto-filling is only done if the number
      of missing entries is <= fill_last, otherwise no filling is done (a data hole is assumed).
    - `show_progress`: whether or not to enable tqdm progress bars within the function (False when running multi-process)

    Output DataFrame format:
    | Field       | Data type  |
    |-------------|------------|
    | `trip_id`   | `str`      |
    | `route_id`  | `int`      |
    | `stop_id`   | `int`      |
    | `stop_name` | `str`      |
    | `date`      | `datetime` |
    """
    trips = {}  # key: trip_id, val: trip data structure built in loop
    
    global LAST_CROSSING_STATS
    LAST_CROSSING_STATS = {
        "total_geoindexed": 0,  # total datapoints that are geoindexed
        "total_extrapolated": 0,  # total datapoints that are linearly extrapolated
        "times_extrapolated": 0,  # times extrapolation is used
    }

    # load in stop_times, tacking on stop lat/lon/name, and indexing on trip for optimization
    stop_times = gtfs_file('stop_times', GTFS_DIR)
    stops = gtfs_file('stops', GTFS_DIR)[
        ['stop_id', 'stop_lat', 'stop_lon', 'stop_name']]

    # load in trips and shapes, used for relative distance algorithm
    gtfs_trips = gtfs_file('trips', GTFS_DIR).set_index('trip_id')
    shapes = gtfs_file('shapes', GTFS_DIR)
    geo_shapes = gpd.GeoDataFrame(shapes, geometry=gpd.points_from_xy(
        shapes.shape_pt_lat, shapes.shape_pt_lon), crs="EPSG:4326")
    geo_shapes = geo_shapes.set_index("shape_id")

    stop_times = stop_times.join(stops.set_index(
        'stop_id'), on='stop_id', how='left')
    # optimization to improve indexing time in loop
    stop_times = stop_times.set_index('trip_id')

    unique_trips = data.trip_id.drop_duplicates()

    # initialize trips structure, separating out stop times, and adding stop location data to stop_times data
    # shape data is also split apart and added here
    for trip_id in tqdm(unique_trips, desc="Initializing trips data structure", disable=(not show_progress)):
        trip_stop_times = stop_times.loc[trip_id]
        trip_shape_id = gtfs_trips.loc[trip_id].shape_id
        shape_data = geo_shapes.loc[trip_shape_id]
        shape_data = shape_data.reset_index()

        trips[trip_id] = {
            # next stop to cross (1 indexed to match stop_sequence)
            'next_stop': 1,
            # list where index is stop_sequence, value is datetime of crossing
            'crossing_times': [None] * (len(trip_stop_times) + 1),
            # Pandas DataFrame of stop times for just this trip (cache to save time)
            'stop_times': trip_stop_times,
            # GeoPandas DataFrame of shape information for just this trip
            'shapes': shape_data,
            # last data point for when a location from this vehicle was reported
            'last_data_point': None,
            'route_id': gtfs_trips.loc[trip_id].route_id
        }

    # main algorithm: loop through all data points, using them to populate crossing times
    for data_point in tqdm(data.itertuples(), desc="Processing vehicle location data", total=len(data), disable=(not show_progress)):
        trip = trips[data_point.trip_id]
        if trip['next_stop'] == -1:
            # this trip already marked complete, skip this data point
            continue

        # report a timepoint, even if we don't have location data
        trip['last_data_point'] = data_point

        if data_point.latitude == 0.0 or data_point.longitude == 0.0:
            # erroneous data point, skip it
            continue

        next_stop = trip['stop_times'][trip['stop_times'].stop_sequence ==
                                       trip['next_stop']].iloc[0]
        next_stop_distance = GD((next_stop.stop_lat, next_stop.stop_lon),
                                (data_point.latitude, data_point.longitude)).miles

        # if first stop, increment if out of visit radius
        if trip['next_stop'] == 1:
            if next_stop_distance > VISIT_RADIUS:
                # don't record a trip as starting more than two minutes early
                dp_date = pd.to_datetime(data_point.date)
                sched_arrival_time = trip['stop_times'].iloc[0].arrival_time
                arrival_split = [int(i) for i in sched_arrival_time.split(":")]
                if arrival_split[0] >= 24:
                    arrival_split[0] -= 24
                if arrival_split[0] < 0:
                    # huh
                    arrival_split[0] += 24
                sched_date = dp_date.replace(
                    hour=arrival_split[0], minute=arrival_split[1], second=arrival_split[2])

                # add or subtract a day if distance between schedule and arrival is super far (over a date split)
                if abs(sched_date - dp_date) > abs((sched_date + timedelta(days=1)) - dp_date):
                    sched_date += timedelta(days=1)
                if abs(sched_date - dp_date) > abs((sched_date - timedelta(days=1)) - dp_date):
                    sched_date -= timedelta(days=1)
                
                if (dp_date - sched_date) >= timedelta(minutes=EARLIEST_START):
                    # only record if data point date is less than two minutes early
                    trip['next_stop'] += 1
                    trip['crossing_times'][1] = data_point.date
                
            continue

        # if last stop, increment only if inside visit radius
        if trip['next_stop'] == len(trip['stop_times']) and next_stop_distance < VISIT_RADIUS:
            trip['crossing_times'][trip['next_stop']] = data_point.date
            trip['next_stop'] = -1  # trip properly completed, do not use again
            continue

        # all other stops, increment if inside visit radius
        if next_stop_distance < VISIT_RADIUS:
            trip['crossing_times'][trip['next_stop']] = data_point.date
            trip['next_stop'] += 1
            continue

        # if here, location data isn't near our stop, but could be after it
        # use geo-indexing to find nearest shape location and determine how far along the route we are
        query = trip['shapes'].sindex.nearest(
            Point(data_point.latitude, data_point.longitude), return_all=False)
        dist_traveled = trip['shapes'].loc[query[1]
                                           ].shape_dist_traveled.iloc[0]

        # for linear extrapolation purposes, determine how many stops we are crossing, and compute the time increase between each one
        # find stops we will cross
        dist_filter = trip['stop_times'][trip['stop_times']
                                         .shape_dist_traveled < dist_traveled]
        # filter out stops we have recorded already
        dist_filter = dist_filter[dist_filter.stop_sequence >=
                                  trip['next_stop']]
        # filter out last stop, since that doesn't use this alg
        dist_filter = dist_filter[dist_filter.stop_sequence < len(
            trip['stop_times'])]

        if len(dist_filter) == 0:
            # no use checking here, no new stops crossed
            continue

        prev_date = pd.to_datetime(
            trip['crossing_times'][trip['next_stop'] - 1])
        target_date = pd.to_datetime(data_point.date)
        delta_t = (target_date - prev_date) / len(dist_filter)

        if len(dist_filter) > MAX_MISSED_STOPS:
            # too large of a gap, skip this
            continue

        # additional sanity checking if we are recording more than one stop crossing
        if len(dist_filter) > 1:
            # sanity check: compute average vehicle speed from this data point and from the schedule, and drop it if over threshold
            # compute dates to match prev & target date
            prev_arrival_time = trip['stop_times'].iloc[trip['next_stop'] - 1].arrival_time
            p_arrival_split = [int(i) for i in prev_arrival_time.split(":")]
            if p_arrival_split[0] >= 24:
                p_arrival_split[0] -= 24
            if p_arrival_split[0] < 0:
                # huh
                p_arrival_split[0] += 24
            prev_sched_date = prev_date.replace(
                hour=p_arrival_split[0], minute=p_arrival_split[1], second=p_arrival_split[2])

            # add or subtract a day if distance between schedule and arrival is super far (over a date split)
            if abs(prev_sched_date - prev_date) > abs((prev_sched_date + timedelta(days=1)) - prev_date):
                prev_sched_date += timedelta(days=1)
            if abs(prev_sched_date - prev_date) > abs((prev_sched_date - timedelta(days=1)) - prev_date):
                prev_sched_date -= timedelta(days=1)

            target_arrival_time = dist_filter.iloc[-1].arrival_time
            t_arrival_split = [int(i) for i in target_arrival_time.split(":")]
            if t_arrival_split[0] >= 24:
                t_arrival_split[0] -= 24
            if t_arrival_split[0] < 0:
                # huh
                t_arrival_split[0] += 24
            target_sched_date = target_date.replace(
                hour=t_arrival_split[0], minute=t_arrival_split[1], second=t_arrival_split[2])

            # add or subtract a day if distance between schedule and arrival is super far (over a date split)
            if abs(target_sched_date - target_date) > abs((target_sched_date + timedelta(days=1)) - target_date):
                target_sched_date += timedelta(days=1)
            if abs(target_sched_date - target_date) > abs((target_sched_date - timedelta(days=1)) - target_date):
                target_sched_date -= timedelta(days=1)

            # now compute speed as (shape distance) / total time
            sched_speed = (dist_filter.iloc[-1].shape_dist_traveled - trip['stop_times'].iloc[trip['next_stop'] -
                        1].shape_dist_traveled) / (target_sched_date - prev_sched_date).total_seconds()
            alg_speed = (dist_filter.iloc[-1].shape_dist_traveled - trip['stop_times'].iloc[trip['next_stop'] -
                        1].shape_dist_traveled) / (target_date - prev_date).total_seconds()
            
            if (alg_speed / sched_speed) > MAX_SPEED_DIFFERENTIAL:
                # too fast! likely an erroneous data point, discard it
                continue

            LAST_CROSSING_STATS["times_extrapolated"] += 1
            LAST_CROSSING_STATS["total_extrapolated"] += len(dist_filter) - 1  # final stop isn't extrapolated here

        # now, mark every stop now and in the future that is earlier than us in the route as visited
        curr_date = prev_date + delta_t
        while next_stop.shape_dist_traveled < dist_traveled and trip['next_stop'] < len(trip['stop_times']):
            trip['crossing_times'][trip['next_stop']] = str(curr_date)
            trip['next_stop'] += 1

            next_stop = trip['stop_times'][trip['stop_times'].stop_sequence ==
                                           trip['next_stop']].iloc[0]

            curr_date += delta_t
            LAST_CROSSING_STATS["total_geoindexed"] += 1

    # finally, convert to DataFrame
    result = {
        'trip_id': [],
        'route_id': [],
        'stop_id': [],
        'stop_name': [],
        'date': []
    }

    for (trip_id, trip_data) in tqdm(trips.items(), desc="Preparing results for DataFrame conversion", disable=(not show_progress)):
        # first index without an entry, may be nothing (+1 to counteract [1:])
        try:
            first_empty = trip_data['crossing_times'][1:].index(None) + 1
        except ValueError:
            first_empty = -1  # doesn't matter, won't ever be tested
        for i in range(1, len(trip_data['crossing_times'])):
            if trip_data['crossing_times'][i] == None:
                # no location data for this point: if `fill_last` is engaged, autofill last time, otherwise say None
                if first_empty >= len(trip_data['crossing_times']) - fill_last:
                    # basically, if the first empty spot is within the last `fill_last` entries, then fill
                    # TODO: In the future, this would be better as some sort of forward extrapolation
                    result['date'].append(trip_data['last_data_point'].date)
                else:
                    result['date'].append(None)
            else:
                # append the recorded crossing time
                result['date'].append(trip_data['crossing_times'][i])

            result['trip_id'].append(trip_id)
            result['route_id'].append(trip_data['route_id'])

            current_stop = trip_data['stop_times'][trip_data['stop_times'].stop_sequence == i].iloc[0]

            result['stop_id'].append(current_stop.stop_id)
            result['stop_name'].append(current_stop.stop_name)

    return pd.DataFrame.from_dict(result)


def calculate_crosses_mult(data: pd.DataFrame,
                           fill_last: int = 0,
                           nodes: int = 4,
                           trip_splits: int = 4) -> pd.DataFrame:
    """
    Run `calculate_crosses` using a ProcessPool to speed things up. See `calculate_crosses` documentation
    for details, same calling convention here.

    Additional parameters for this function:
    - `nodes`: number of Pool nodes
    - `trip_splits`: number of splits of `data` to make
    """
    pool = ProcessPool(nodes=nodes)

    unique_trips = list(data.trip_id.drop_duplicates())

    def data_generator():
        chunk_size = int(len(unique_trips) / trip_splits)
        for i in range(0, trip_splits):
            if i == trip_splits - 1:
                trips = unique_trips[chunk_size*i:]
            else:
                trips = unique_trips[chunk_size*(i):chunk_size*(i+1)]
            yield data[data.trip_id.isin(trips)]

    def map_function(data_split: pd.DataFrame):
        return calculate_crosses(data_split, fill_last=fill_last, show_progress=False)

    return pd.concat(
        tqdm(
            pool.imap(map_function, data_generator()),
            desc="Calculating batched crossing data",
            total=trip_splits
        )
    )


def append_schedule(crossing_data: pd.DataFrame) -> pd.DataFrame:
    """
    Append the scheduled arrival time as the string column `arrival_time` from stop_times GTFS
    to the crossing data in `crossing_data`.
    """
    stop_times = gtfs_file('stop_times', GTFS_DIR)
    stop_times = stop_times[['trip_id', 'stop_id', 'arrival_time']]
    return crossing_data.merge(stop_times, on=['trip_id', 'stop_id'], how="inner")
