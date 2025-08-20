"""
window: More detailed evaluation metrics for window_gnn models. Useful metrics:
- Average MAE by time of day
- Average MAE by difficulty of prediction (how far in the future it is)
- Standard deviation of predictions (by time of day, or difficulty)
- Similar stats, across days

Author: Nick Gable (gable105@umn.edu)
"""

from realtime_transit.models.window_gnn import *
import torch
from tqdm import tqdm
from typing import List, Tuple

OUTPUT_DIR = "data/output"


def error_tensor(model: ModelS, test_set: WindowDataset) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Returns a list of compact error tensors containing error values propagated across all predictions and all times, as well as a
    similarly formatted list of tensors representing the absolute expected predicted time.

    Dimensions: (day)[t val, stop, windowed trips]

    When both sets of tensors are available, this should result in easy and repeatable computations. The index of time values
    in the time map is used as t_val instead of the actual time (so index 0 is the data for the first time value).

    For additional analysis, this function returns a third set of tensors scheduled_vals that reports the scheduled data, linked
    in a similar manner to the first two sets.

    Output: (errors, expected_vals, scheduled_vals)
    """
    model.eval()
    output = [torch.zeros((len(test_set.time_vals), len(
        test_set[0].x), int(test_set.data[0].t_max_2)), dtype=torch.float32).to_sparse() for _ in range(len(test_set.data))]
    expected_vals = [torch.zeros(output[0].size(), dtype=torch.float32).to_sparse() for _ in range(len(test_set.data))]
    scheduled_vals = [torch.zeros(output[0].size(), dtype=torch.float32).to_sparse() for _ in range(len(test_set.data))]

    current_day = 0
    output[0] = output[0].to_dense()
    expected_vals[0] = expected_vals[0].to_dense()
    scheduled_vals[0] = scheduled_vals[0].to_dense()

    # used to provide absolute time of day information
    test_set_absolute = WindowDataset(test_set.data, test_set.time_vals, test_set.type_2)

    t_max = test_set.data[0].t_max_2
    for i in tqdm(range(len(test_set)), desc="Computing error over test set"):
        day, time_val = test_set.idx_map[i]
        if current_day != day:
            output[current_day] = output[current_day].to_sparse()
            expected_vals[current_day] = expected_vals[current_day].to_sparse()
            scheduled_vals[current_day] = scheduled_vals[current_day].to_sparse()
            
            output[day] = output[day].to_dense()
            expected_vals[day] = expected_vals[day].to_dense()
            scheduled_vals[day] = scheduled_vals[day].to_dense()

            current_day = day

        windowed_data = test_set[i]

        # apply transformations if the model needs it
        if model.args.no_trip_id:
            # remove trip ID half of the input vector
            windowed_data.x = windowed_data.x[:, t_max:]
        if model.args.no_schedule and not model.args.no_trip_id:
            # remove schedule portion of input vector
            windowed_data.x = windowed_data.x[:, :2*t_max]
        if model.args.no_schedule and model.args.no_trip_id:
            # only send in the known time values
            windowed_data.x = windowed_data.x[:, t_max:2*t_max]

        predicted = model(windowed_data)
        target = windowed_data.y

        if model.args.schedule_deviation and model.args.normalize:
            # want to de-normalize these values so that the loss values in testing are meaningful
            selector = predicted[target != 0]
            predicted[target != 0] = (
                selector * 0.010416) - 0.000347

            selector = target[target != 0]
            target[target != 0] = (
                selector * 0.010416) - 0.000347

        # compute error for each item, add to our output
        predicted[target == 0] = 0
        error = target-predicted

        output[day][int(time_val * (86400 / 300))].copy_(error.data)

        # also compute absolute expected values for each item, add to our output
        expected_vals[day][int(time_val * (86400 / 300))].copy_(test_set_absolute[i].y.data)

        # also return scheduled values in a similar manner
        scheduled_vals[day][int(time_val * (86400 / 300))].copy_(test_set_absolute[i].x[:, 2*t_max:])

    output[current_day] = output[current_day].to_sparse()
    expected_vals[current_day] = expected_vals[current_day].to_sparse()
    scheduled_vals[current_day] = scheduled_vals[current_day].to_sparse()

    return (output, expected_vals, scheduled_vals)


if __name__ == "__main__":
    import argparse
    import os
    from realtime_transit.pipeline import *

    parser = argparse.ArgumentParser(
        "window.py", description="Evaluator for window_gnn models.")
    parser.add_argument("file", type=str, help="Model to evaluate")
    parser.add_argument("--train-set", action="store_true", default=False, help="Use training set instead of testing set")

    args = parser.parse_args()

    cuda_init()
    if torch.cuda.is_available():
        map_location = torch.device('cuda')
    else:
        map_location = torch.device('cpu')

    model = torch.load(os.path.join(
        OUTPUT_DIR, f"{args.file}.pt"), map_location=map_location)
    if "hf" in args.file:
        input_file_name = "input_hf_network_windowed_scheduled.pt"
    else:
        input_file_name = "input_small_network_windowed_scheduled.pt"

    input_data = torch.load(os.path.join(
        OUTPUT_DIR, input_file_name), map_location=map_location)
    
    test_set = WindowDataset((input_data[14:] if not args.train_set else input_data[:14]), torch.arange(0, 1, 300/86400), type_2=True,
                             schedule_deviation=model.args.schedule_deviation, normalize=model.args.normalize, outlier_cut=model.args.outlier_cut)
    
    (error, expected, scheduled) = error_tensor(model, test_set)

    if args.train_set:
        args.file += "-train"

    output_file_name = f"{args.file}-error.pt"
    print(f"Saving error tensor as {output_file_name}")
    torch.save(error, os.path.join(OUTPUT_DIR, output_file_name))

    output_file_name = f"{args.file}-expected.pt"
    print(f"Saving expectation tensor as {output_file_name}")
    torch.save(expected, os.path.join(OUTPUT_DIR, output_file_name))

    output_file_name = f"{args.file}-scheduled.pt"
    print(f"Saving schedule tensor as {output_file_name}")
    torch.save(scheduled, os.path.join(OUTPUT_DIR, output_file_name))
