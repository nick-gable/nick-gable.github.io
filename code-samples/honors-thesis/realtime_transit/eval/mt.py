"""
mt: Evaluate performance of Metro Transit predictions in comparison to our own.

Author: Nick Gable (gable105@umn.edu)
"""

import pandas as pd
from typing import Tuple
from tqdm import tqdm


def compute_visits(data: pd.DataFrame) -> pd.DataFrame:
    """
    Given a day of prediction data, return a DataFrame containing the best estimate of actual stop crossing times
    for each trip at each stop.

    Rule: A stop crossing is reported as the last date that a trip/stop crossing pairing appears in the feed, discounting
    feed reports that report an early past departure (which are kept in the feed to report early departures instead of having
    no data).
    """
    # convert to datetime columns
    data.date = pd.to_datetime(data.date)
    data.predicted_departure = pd.to_datetime(data.predicted_departure)

    # filter out past departure reports
    data = data[data.date <= data.predicted_departure]

    # group by trip and stop ID, and get max date values
    data = data[['trip_id', 'stop_id', 'date']]
    return data.groupby(['trip_id', 'stop_id']).max().reset_index()


def compute_deviations(day_data: pd.DataFrame, visit_data: pd.DataFrame) -> pd.DataFrame:
    """ 
    Given a full day of prediction data, as well as visit data from `compute_visits`, compute the deviations between the
    likely departure time of the trip and other predictions made. 
    """
    visit_data = visit_data.rename(columns={'date': 'actual_departure'})
    day_data = day_data.merge(
        visit_data, on=['trip_id', 'stop_id'], how='right')
    day_data['deviation'] = day_data.actual_departure - \
        day_data.predicted_departure

    return day_data[['trip_id', 'stop_id', 'predicted_departure', 'actual_departure', 'deviation']]


def compute_loss(deviation_data: pd.DataFrame) -> Tuple[float, float]:
    """
    Given prediction deviation data, return the (MSE,MAE) tuple in the same units as the test loop for the window_gnn model.
    """
    mae = deviation_data.deviation.abs().sum() / len(deviation_data)
    mae = mae.total_seconds() / 86400
    mse = (deviation_data.deviation.dt.total_seconds() /
           86400).pow(2).sum() / len(deviation_data)

    return mse, mae


if __name__ == "__main__":
    """Fun time: Compute loss across all days"""
    SN_ROUTES = [2, 3, 6, 113, 114, 902]
    HF_ROUTES = [2,3,6,10,11,18,21,54,64,63,921,923,924,901,902,904]

    from realtime_transit.pipeline import *

    mae_vals_sn = []
    mse_vals_sn = []
    mae_vals_hf = []
    mse_vals_hf = []
    
    pred_data = get_collected_data('pred', start_date=COLLECTION_START+timedelta(days=14))

    for day_data in tqdm(pred_data, total=7, desc="Computing MT loss"):
        filtered_data = day_data[day_data.route_id.isin(SN_ROUTES)]
        filtered_data = filtered_data.loc[:]
        visits = compute_visits(filtered_data)
        deviations = compute_deviations(filtered_data, visits)
        # filter out likely erroneous deviations that are very large
        deviations = deviations[deviations.deviation.abs() < timedelta(minutes=30)]
        mse,mae = compute_loss(deviations)
        mae_vals_sn.append(mae)
        mse_vals_sn.append(mse)

        filtered_data = day_data[day_data.route_id.isin(HF_ROUTES)]
        filtered_data = filtered_data.loc[:]
        visits = compute_visits(filtered_data)
        deviations = compute_deviations(filtered_data, visits)
        deviations = deviations[deviations.deviation.abs() < timedelta(minutes=30)]
        mse,mae = compute_loss(deviations)
        mae_vals_hf.append(mae)
        mse_vals_hf.append(mse)

    print("---Results---")
    print("---HF network---")
    print(f"MAE vals: {mae_vals_hf}")
    print(f"MSE vals: {mse_vals_hf}")
    print(f"Average MAE: {sum(mae_vals_hf) / len(mae_vals_hf)}")
    print(f"Average MSE: {sum(mse_vals_hf) / len(mse_vals_hf)}")

    print("---Small network---")
    print(f"MAE vals: {mae_vals_sn}")
    print(f"MSE vals: {mse_vals_sn}")
    print(f"Average MAE: {sum(mae_vals_sn) / len(mae_vals_sn)}")
    print(f"Average MSE: {sum(mse_vals_sn) / len(mse_vals_sn)}")
