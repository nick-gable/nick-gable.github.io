"""
end_route_mlp: MLP based model used to predict the arrival time for the last stop along a single route.

This model isn't very good: it was used as a test model so that I could get used to TensorFlow. A few 
improvements that may be useful for other models:
- Parallelize data preprocessing
- Move GD calculations to earlier in the preprocessing chain when there would be less calculations
- Better strategy for determining when a vehicle is at a stop
- Better training procedures
- Better model design

(You get the idea. Not an exhaustive list.)

Author: Nick Gable (gable105@umn.edu)
"""

import pandas as pd
from geopy.distance import geodesic as GD
import os
import logging
import tensorflow as tf
import numpy as np

ROUTE = 3
DIR = "WB"
BACK_AMOUNT = 5  # number of data points back to feed into model
# km distance to be considered "at" a stop: this is roughly one long block
VISIT_RADIUS = 0.25
TEST_SIZE = 0.2  # size of test dataset

logger = logging.getLogger("end_route_mlp")
logging.basicConfig(level=logging.DEBUG)


def gtfs_file(file: str, gtfs_dir: str):
    return pd.read_csv(os.path.join(gtfs_dir, f"{file}.txt"))


def preprocess_data(file: str, gtfs_dir: str) -> pd.DataFrame:
    """
    Preprocess raw `rt-pos` data from `file` for training this model. Use GTFS records in `gtfs_dir`
    to do so.
    """
    logger.info("Starting end_route_mlp data pre-process")

    # load in relevant data files
    gps_data = pd.read_hdf(file)
    stop_times = gtfs_file("stop_times", gtfs_dir)
    stops = gtfs_file("stops", gtfs_dir)
    trips = gtfs_file("trips", gtfs_dir)

    # filter down GPS data to only include dir and route
    avail_trips = trips[trips.route_id == ROUTE]  # only selected route
    # only selected direction
    avail_trips = avail_trips[trips.direction == DIR]

    gps_data = gps_data.merge(avail_trips.trip_id, how="inner", on="trip_id")

    # find target last stop we are aiming for
    # example trip_id to filter on
    trip_id = gps_data.trip_id.drop_duplicates().iloc[0]
    filtered_times = stop_times[stop_times.trip_id == trip_id]
    max_seq = filtered_times.stop_sequence.max()
    last_stop_id = int(
        filtered_times[filtered_times.stop_sequence == max_seq].stop_id.iloc[0])
    last_stop = stops[stops.stop_id == last_stop_id]

    # filter stops by distance from terminal stop loc
    stop_lat = last_stop.iloc[0].stop_lat
    stop_lon = last_stop.iloc[0].stop_lon

    gps_data['distance'] = gps_data.apply(
        lambda row: GD((stop_lat, stop_lon),
                       (row['latitude'], row['longitude'])),
        axis=1
    )

    gps_data = gps_data.sort_values('distance')

    def transform_trip(trip: pd.DataFrame):
        """
        Helper function used on entire collection of data from trip, transforming it into
        desired output.
        """
        if trip.distance.min() > VISIT_RADIUS:
            return pd.DataFrame([])  # no record of this trip finishing

        trip = trip.sort_values('distance', ascending=False)
        trip = trip[["latitude", "longitude", "distance", "date"]]

        # stop immediately once we have crossed visit radius
        stop_distance = trip[trip.distance <= VISIT_RADIUS].distance.max()
        arrival_time = trip[trip.distance == stop_distance].date.iloc[0]

        results_dict = {}

        for i in range(BACK_AMOUNT):
            results_dict[f'lat-{i}'] = []
            results_dict[f'lon-{i}'] = []

        results_dict['time_left'] = []

        for i in range(len(trip)):
            for j in range(BACK_AMOUNT):
                if i-j < 0:
                    results_dict[f'lat-{j}'].append(0.0)
                    results_dict[f'lon-{j}'].append(0.0)
                else:
                    results_dict[f'lat-{j}'].append(trip.iloc[i-j].latitude)
                    results_dict[f'lon-{j}'].append(trip.iloc[i-j].longitude)

            results_dict['time_left'].append(
                (arrival_time - trip.iloc[i].date))
            if trip.iloc[i].distance == stop_distance:
                break

        return pd.DataFrame.from_dict(results_dict)

    result = gps_data.groupby(['trip_id']).apply(transform_trip)

    # drop index from group by, convert time left into minutes
    result = result.reset_index(drop=True)
    result.time_left = result.time_left.apply(lambda a: a.total_seconds()) / 60

    # convert latitude / longitude data into stop_seq / distance data
    filtered_stops = filtered_times.merge(
        stops[['stop_lat', 'stop_lon', 'stop_id']], how="inner", on="stop_id")
    filtered_stops = filtered_stops[['stop_sequence', 'stop_lat', 'stop_lon']]
    filtered_stops

    def to_stop_seq(record: pd.Series):
        """
        Helper function to convert latitude / longitude raw data into closest stop_seq / distance
        """
        stops = filtered_stops
        for i in range(BACK_AMOUNT):
            lat, lon = record[f'lat-{i}'], record[f'lon-{i}']
            if abs(lat) < 0.001 or abs(lon) < 0.001:
                record[f'seq-{i}'] = -1
                record[f'dist-{i}'] = -1
                continue
            stops['distance'] = stops.apply(lambda row: GD(
                (lat, lon), (row['stop_lat'], row['stop_lon'])), axis=1)
            closest_stop = stops[stops.distance == stops.distance.min()]
            record[f'seq-{i}'] = int(closest_stop.stop_sequence.iloc[0])
            record[f'dist-{i}'] = float(closest_stop.distance.iloc[0].km)

        return record.drop([f'lat-{i}' for i in range(BACK_AMOUNT)] + [f'lon-{i}' for i in range(BACK_AMOUNT)])

    result = result.apply(to_stop_seq, axis=1)

    return result


def train_model(data: pd.DataFrame) -> tf.keras.models.Sequential:
    """
    Train the `end_route_mlp` model on the provided pre-processed data. Returns
    model after training.
    """
    # assemble training and testing datasets
    np.random.seed(159)
    test_mask = np.random.rand(len(data)) < TEST_SIZE
    train = data[~test_mask].reset_index(drop=True)
    test = data[test_mask].reset_index(drop=True)

    x_train = train.drop('time_left', axis=1)
    y_train = train['time_left']
    x_test = test.drop('time_left', axis=1)
    y_test = test['time_left']
    train[train['seq-0'] == 60.0]

    # set up model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(40, input_shape=(x_train.shape[1],)),
        tf.keras.layers.AlphaDropout(10),
        tf.keras.layers.Dense(1)
    ])

    loss_fn = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer='adam', loss=loss_fn, metrics=['mae'])

    # train model
    model.fit(x_train, y_train, epochs=30)
    model.evaluate(x_test, y_test, verbose=2)

    return model
