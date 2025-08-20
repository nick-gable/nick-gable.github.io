"""
window_gnn: GNN model using a "window" strategy for feeding in data, training, and predictions.

Window strategy: \
1. At a given time, select all "active" trips, i.e. trips where the vehicle is between its first or last stops,
(with some sort of buffer so that first and last stop arrivals are predicted). Assign these trips index values starting from 0.
2. For each trip, record in the input feature vector a) an array of trips that visit this stop within the window (including future visits),
and b) a same-length array of known visits to this stop at this time, with the indices in this array mapping to the previously mentioned array of trips.
3. For output features, have one vector with a format similar to the array in step 2b above, containing all the predictions within this window.

Author: Nick Gable (gable105@umn.edu)
"""
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, Linear, SAGEConv, SAGPooling, GraphConv, GCN
from torch.nn import L1Loss, MSELoss
from torch.optim.lr_scheduler import ExponentialLR, CyclicLR
import torch.nn.functional as F
from realtime_transit.graphs import TransitNetwork, TransitStop
from typing import Iterator, Union, List
from random import shuffle
import math
import copy
from tqdm import tqdm

import argparse


class WindowData:
    """
    Helper class used to contain metadata about a dataset so that windowed data can be quickly generated for model training.

    Important note: The way this class is implemented prevents the very first trip from the dataset (which is assigned trip index 0)
    from ever being added to a window. To resolve this problem, make sure the data fed into this class contains a dummy 0th trip that does
    not actually represent a real trip in the network (one option being to set the start_time to 2 (never reached) and end time to -1 (never reached)).
    """

    def __init__(self,
                 start_times: torch.Tensor,
                 end_times: torch.Tensor,
                 node_visits: torch.Tensor,
                 node_visit_times: torch.Tensor,
                 scheduled_visit_times: torch.Tensor,
                 edge_index: torch.Tensor):
        """
        Let `n_t` be the total number of trips in the system, `t_max` be the maximum number of trips visiting any stop during any given window,
        and `n_s` be the total number of stops in the system.

        Params:
        `start_times`: `n_t` size vector, with float values between (0,1) representing when a trip at the given index should start in a window.
        `end_times`: `n_t` size vector, same format as `start_times` but for when a trip should leave the window.
        `node_visits`: `[n_s, n_t]` size vector, with each row being a node feature vector that contains a boolean 0/1 array of n_t size
        indicating whether the trip at that index visits the stop.
        `node_visit_times`: `[n_s, n_t]` size vector, where each index represents the (0,1) visit time of that trip at the stop, or is zero if no
        visit occurs.
        `scheduled_visit_times`: `[n_s, n_t]` size vector, like `node_visit_times` but the scheduled instead of actual visit time.
        `edge_index`: Edge tensor connecting nodes to each other, as is typical for PyG.
        """
        self.start_times = start_times
        self.end_times = end_times
        self.node_visits = node_visits
        self.node_visit_times = node_visit_times
        self.scheduled_visit_times = scheduled_visit_times
        self.edge_index = edge_index

        # compute initial t_max
        # TODO: Revise: this is too big, since it counts maximum trips throughout the day when we want the size when the window
        # is the biggest. This also should be computed across the entire dataset (beyond a single instance of this class) which
        # will take a bit more effort.
        self.t_max = int(
            torch.sum(  # sum up vector of True/False values (0 or 1) for each row
                self.node_visits != 0,
                dim=1  # row wise
            ).max()  # maximum is our t_max
        )

        self.n_t = len(self.start_times)

    def get_windowed_data(self, time: float) -> Data:
        """
        Generate a Data object used to train the model at the given time. Time should be a float between 0 and 1 representing
        the current time relative to the 4 AM starting transit day (so 0 = 4 AM at the start, 1 = 4 AM at the end).
        """
        # binary mask indicating if the trip at that index should be in the window
        trip_mask = (self.start_times <= time).int() * \
            (self.end_times >= time).int()

        # use trip mask to mask visits
        masked_visits = self.node_visits * trip_mask
        # convert mask into index list
        trip_indices = masked_visits * torch.arange(self.n_t)

        # justify elements in trip_indices
        # TODO: Currently uses a sort - can we find something more efficient?
        # since it is only sorting 0s and 1s though, might be fast enough
        index = torch.sort((trip_indices != 0).int(),
                           dim=1, descending=True)[1]
        trip_indices = trip_indices.gather(1, index)

        # now, use trip_indices to compute visit times
        visit_times = self.node_visit_times.gather(
            1, trip_indices.type(torch.LongTensor).to(trip_indices.device))

        # next, narrow down trip_indices and visit times to t_max length
        trip_indices = trip_indices.narrow(dim=1, start=0, length=self.t_max)
        visit_times = visit_times.narrow(dim=1, start=0, length=self.t_max)

        # now, split visit_times into input (where times are <= time) and output (times > time)
        # TODO: May want an easy toggle for this filtering in case it is actually helpful for the model to make these predictions
        visit_times_output = torch.clone(visit_times)
        visit_times[visit_times > time] = 0  # predict these
        # existing values, no need to predict
        visit_times_output[visit_times_output <= time] = 0

        # finally, concat values, create Data object, and return
        return Data(x=torch.cat((trip_indices, visit_times), dim=1), y=visit_times_output, edge_index=self.edge_index)

    def get_windowed_data_2(self, time: float, only_pred_schedule: bool = False, schedule_deviation: bool = False) -> Data:
        """
        Generate a Data object used to train the model at the given time. Time should be a float between 0 and 1 representing
        the current time relative to the 4 AM starting transit day (so 0 = 4 AM at the start, 1 = 4 AM at the end).

        Note: This modified "type 2" window does not output trip ID number vectors, and uses a larger `t_max` (`t_max_2`) so value so that
        trip indices are consistent across all nodes. Make sure that `t_max_2` has been set since this is not computed at all in this element.

        Optional parameter `only_pred_schedule` zeros out all schedule values except those that are related to future predictions (nodes we desire
        the model to predict). `schedule_deviation` sets time values in `x` and `y` as (actual-schedule) instead of the absolute time.
        """
        # binary mask indicating if the trip at that index should be in the window
        trip_mask = (self.start_times <= time).int() * \
            (self.end_times >= time).int()

        # # use trip mask to mask visits
        # masked_visits = self.node_visits * trip_mask

        # # type 2 change: sum all row vectors together, getting a single combined mask of all trips in the window
        # masked_visits = masked_visits.sum(dim=0)
        # masked_visits = (masked_visits != 0).int()  # back to 0/1 mask for simplicity

        # convert mask into index list
        trip_indices = trip_mask * torch.arange(self.n_t)

        # justify elements in trip_indices
        index = torch.sort((trip_indices != 0).int(),
                           dim=0, descending=True)[1]
        trip_indices = trip_indices.gather(0, index)

        # now, use trip_indices to compute visit times and scheduled visit times
        trip_indices_repeat = trip_indices.type(torch.LongTensor).to(
            trip_indices.device).repeat(len(self.node_visit_times), 1)
        visit_times = self.node_visit_times.gather(1, trip_indices_repeat)

        # for now, no filtering, so this will include trip visit times outside of the window (we'll see if that's helpful)
        scheduled_visit_times = self.scheduled_visit_times.gather(
            1, trip_indices_repeat)

        # also use trip_indices to compute trip selection
        trip_selector = self.node_visits.gather(1, trip_indices_repeat)

        # next, narrow down visit times and trip selector to t_max_2 length
        trip_selector = trip_selector.narrow(
            dim=1, start=0, length=self.t_max_2)
        visit_times = visit_times.narrow(dim=1, start=0, length=self.t_max_2)
        scheduled_visit_times = scheduled_visit_times.narrow(
            dim=1, start=0, length=self.t_max_2)

        # mask out trip_selector values so that it is only trips we are predicting
        trip_selector[visit_times <= time] = 0

        # now, split visit_times into input (where times are <= time) and output (times > time)
        visit_times_output = torch.clone(visit_times)
        visit_times[visit_times > time] = 0  # predict these
        # existing values, no need to predict
        visit_times_output[visit_times_output <= time] = 0

        if schedule_deviation:
            # offset visit times as deviations from schedule, and predictions as well
            schedule_offset = scheduled_visit_times * (visit_times != 0).int()
            visit_times -= schedule_offset

            schedule_offset = scheduled_visit_times * trip_selector
            visit_times_output -= schedule_offset

        if only_pred_schedule:
            # set to zero all schedule values that are not related to values we want to predict
            scheduled_visit_times[trip_selector == 0] = 0

        # finally, concat, create Data object, and return
        return Data(x=torch.cat((trip_selector, visit_times, scheduled_visit_times), dim=1), y=visit_times_output, edge_index=self.edge_index)

    def __repr__(self):
        return f"WindowData, n_t={len(self.start_times)}, n_s={len(self.node_visits)}, t_max={self.t_max}"

    def move_to(self, device: str):
        """
        Move all tensors stored in this WindowData object to the specified TF device.
        """
        self.start_times = self.start_times.to(device)
        self.end_times = self.end_times.to(device)
        self.node_visits = self.node_visits.to(device)
        self.node_visit_times = self.node_visit_times.to(device)
        self.scheduled_visit_times = self.scheduled_visit_times.to(device)
        self.edge_index = self.edge_index.to(device)


class WindowDataset():
    """
    Special dataset object that takes in a list of WindowData objects, as well as a range of data to store for them,
    and creates an indexable object that can also be shuffled.
    """

    def __init__(self, data: List[WindowData], time_vals: List[float], type_2: bool = False, only_pred_schedule: bool = False, schedule_deviation: bool = False, normalize: bool = False, outlier_cut: Union[float, None] = None):
        """
        Create a WindowDataset, using `data` as the datasource, across the time values `time_vals`. Optionally, 
        enable `type_2` windowing when generating values from this dataset.

        Use `outlier_cut` when schedule deviation is on to cut out values that deviate more than `outlier_cut` from the ground truth set
        (range 0,1).
        """
        self.data = data
        self.time_vals = time_vals

        # used to map an index value to a specific data point in the dataset
        self.idx_map = [(day, time_val) for day in range(len(self.data))
                        for time_val in self.time_vals]

        self.type_2 = type_2
        self.only_pred_schedule = only_pred_schedule
        self.schedule_deviation = schedule_deviation
        # set to true later, after we access the un-normalized data to compute mean/std
        self.normalize = False

        self.outlier_cut = outlier_cut

        if outlier_cut:
            assert self.schedule_deviation, "Outlier cut not implemented without schedule deviation"

        if normalize:
            # this process skipped in favor of manual rough normalization
            # # calculate mean and std for this dataset
            # assert self.type_2, "Normalization only supported for type 2 window"
            # t_max = self.data[0].t_max_2

            # mean = torch.mean(self[0].x[t_max:], dim=0)
            # std = torch.std(self[0].x[t_max:], dim=0)

            # for i in tqdm(range(1, len(self)), desc="Computing mean/std for normalization"):
            #     selector = self[i].x[t_max:]
            #     selector = selector[selector != 0]
            #     mean += torch.mean(selector, dim=0)
            #     std += torch.std(selector, dim=0)

            # self.mean = mean / len(self)
            # self.std = std / len(self)  # TODO: This might not work, if so compute manually lol (two passes)

            self.normalize = True

    def __len__(self) -> int:
        return len(self.idx_map)

    def __getitem__(self, idx) -> Data:
        # create and return a sliced WindowDataset if a slice is requested
        if type(idx) == slice:
            new_dataset = WindowDataset(self.data, self.time_vals, self.type_2,
                                        self.only_pred_schedule, self.schedule_deviation, self.normalize, self.outlier_cut)
            new_dataset.idx_map = self.idx_map[idx]
            return new_dataset

        (day, time_val) = self.idx_map[idx]
        if self.type_2:
            data = self.data[day].get_windowed_data_2(
                time_val, self.only_pred_schedule, self.schedule_deviation)

            t_max = self.data[day].t_max_2

            if self.outlier_cut:
                # filter out outliers from output dataset
                data.y[data.y > self.outlier_cut] = 0
                data.y[data.y < -self.outlier_cut] = 0

                # remove these from trip selector too to avoid confusing model
                data.x[:, :t_max][data.y == 0] = 0

            if self.normalize and not self.schedule_deviation:
                selector = data.x[:, t_max:]
                selector = selector[selector != 0]

                selector = (selector - 0.5) * 2  # move from (0,1) to (-1, 1)

            if self.normalize and self.schedule_deviation:
                t_max = self.data[day].t_max_2
                selector = data.x[:, t_max:2*t_max]

                # TODO: if this is worthwhile, switch away from hardcoded values
                # normalize deviated values (assumes an average of 30 seconds behind, in the range (-15 min, 15 min))
                selector[selector != 0] = (
                    selector[selector != 0] + 0.000347) / 0.010416

                # do this for output values too
                selector = data.y
                selector[selector != 0] = (
                    selector[selector != 0] + 0.000347) / 0.010416

                # normalize schedule normally
                selector = data.x[:, 2*t_max:]
                selector[selector != 0] = (selector[selector != 0] - 0.5) * 2

            return data
        else:
            return self.data[day].get_windowed_data(time_val)

    def shuffle(self):
        shuffle(self.idx_map)


class WindowCacheDataset(WindowDataset):
    """Extension of `WindowDataset` that caches items by day in CPU/main memory space to save CUDA memory.
    A consequence of this is that data will still be trained in day chunks, although the order of the days 
    as well as the data within those days will be shuffled. Reading from this dataset in any pattern other than
    linear access will have extremely poor performance and likely overload CUDA. Also, obviously don't use this
    if not on a CUDA enabled machine.

    Data passed into this should already be in CPU space (presumably because there isn't enough CUDA memory otherwise)"""

    def __init__(self, data: List[WindowData], time_vals: List[float], type_2: bool = False, only_pred_schedule: bool = False):
        super().__init__(data, time_vals, type_2, only_pred_schedule)

    def __getitem__(self, idx) -> Data:
        (day, _) = self.idx_map[idx]

        # move day we are accessing to CUDA
        self.data[day].move_to('cuda')

        # move previously accessed day (assuming linear access pattern) to CUDA
        prev_day = day - 1
        if day == -1:
            prev_day = len(self.data) - 1

        self.data[prev_day].move_to('cpu')

        # now, actually get the item
        return super().__getitem__(idx)

    def shuffle(self):
        # assume last day was accessed on CUDA in current order, move it to CPU
        self.data[self.idx_map[-1][0]].move_to('cpu')

        # shuffle, but cluster index map so that days are all contiguous, but randomized
        idx_submap = [[time_val for time_val in self.time_vals]
                      for _ in range(len(self.data))]  # one sublist per day

        # shuffle each days list
        for sublist in idx_submap:
            shuffle(sublist)

        # make list of days in order, then shuffle it
        days = list(range(len(self.data)))
        shuffle(days)

        # reconstruct index map, converting to single tuple list
        self.idx_map = [(day, time_val)
                        for day in days for time_val in idx_submap[day]]


def to_data(network: TransitNetwork, num_trips: Union[int, None] = None) -> WindowData:
    """
    Convert TransitNetwork high level information into the graph representation needed
    for this model. Optionally, set number of trips across the system so that it is consistent if 
    there is concern about variable trip counts day-to-day.
    """
    edges = []  # list of (source node, dest node) tuples

    # initialize needed tensors
    # note: add one extra trip so that trip with index 0 is a dummy trip (see WindowData docstring)
    if not num_trips:
        num_trips = len(network.trips) + 1
    else:
        num_trips += 1  # add one to the trip count to account for dummy first trip

    start_times = torch.zeros(num_trips)
    end_times = torch.zeros(num_trips)
    node_visits = torch.zeros(
        (len(network.stops), num_trips), dtype=torch.int32)
    node_visit_times = torch.zeros(
        (len(network.stops), num_trips))
    scheduled_visit_times = torch.zeros(
        (len(network.stops), num_trips)
    )

    start_times[0] = 2  # dummy trip 'starts' after day completes
    end_times[0] = -1  # dummy trip 'ends' before day starts

    # map trip ID values to indicies starting from one (idx zero should not be mapped to a trip)
    trip_map = {trip_id: (idx+1) for (idx, trip_id) in enumerate(
        network.trips)}  # key: trip_id, value: trip index

    # "count out" stops so that they are zero indexed
    stop_indices = {}  # key: stop_id, value: graph index (0 indexed)
    current_idx = 0
    for stop_id in network.stops.keys():
        stop_indices[stop_id] = current_idx
        current_idx += 1

    # enumerate through stops, generating edge and node lists
    for stop_id, stop in tqdm(network.stops.items(), desc="Populating edge list"):
        stop: TransitStop = stop

        # edge enumeration
        for edge in stop.outgoing_edges:
            edges.append((stop_indices[stop.stop_id],
                         stop_indices[edge]))

        # node enumeration
        for trip_id, time in stop.visits.items():
            (hour, minute, scheduled_hour, scheduled_minute) = time
            if math.isnan(hour) or math.isnan(minute):
                seconds = 0  # no crossing recorded, but should have had a crossing
            else:
                hour = hour if hour > 3 else hour + 12  # move first three hours to end of day
                seconds = 3600*(hour-4) + 60*minute

            scheduled_hour = scheduled_hour if scheduled_hour > 3 else scheduled_hour + 12
            scheduled_seconds = 3600*(scheduled_hour-4) + 60*scheduled_minute

            # normalize to range (0,1)
            visit_time = seconds / 86400
            scheduled_visit_time = scheduled_seconds / 86400

            stop_idx = stop_indices[stop_id]
            trip_idx = trip_map[trip_id]

            # populate visit tensors
            node_visits[stop_idx, trip_idx] = 1
            node_visit_times[stop_idx, trip_idx] = visit_time
            scheduled_visit_times[stop_idx, trip_idx] = scheduled_visit_time

            # update start and end times if appropriate
            if start_times[trip_idx] == 0 or start_times[trip_idx] > visit_time:
                start_times[trip_idx] = visit_time

            if end_times[trip_idx] < visit_time:
                end_times[trip_idx] = visit_time

    # build WindowData object and return
    edge_index = torch.tensor(edges, dtype=torch.int64)
    return WindowData(start_times, end_times, node_visits, node_visit_times, scheduled_visit_times, edge_index.t().contiguous())


def get_optimized_t_max(dataset: List[WindowData]) -> int:
    """
    Based off of the dataset contained in `dataset`, compute the lowest possible `t_max` value and return it.
    """
    optimal_t_max = 0

    # compute every 5 minutes (matches training loop)
    time_range = torch.arange(0, 1, 300/86400)
    num_timepoints = len(time_range)
    # assumes that all days have same number of trips (they should)
    num_trips = len(dataset[0].start_times)
    # expands into a matrix `num_timepoints` x `num_trips`
    time_range = time_range.repeat(num_trips, 1).t()

    for day in tqdm(dataset, desc="Computing optimal t_max value"):
        # duplicate start and end times by num_timepoints
        start_times = day.start_times.repeat(num_timepoints, 1)
        end_times = day.end_times.repeat(num_timepoints, 1)

        # now that dimensions match, compute all windowed times throughout the day
        start_times = (start_times <= time_range).int()
        end_times = (end_times >= time_range).int()

        # multiply together to compute windows
        combined_times = start_times * end_times

        # because of space constaints, need to iterate this
        for window in tqdm(combined_times, desc="Iterating by day", leave=False):
            # apply window mask
            node_visits = day.node_visits * window

            # compute non-zero value counts, take max
            t_max = int(
                torch.sum(  # sum up vector of True/False values (0 or 1) for each row
                    node_visits != 0,
                    dim=1  # row wise
                ).max()  # maximum is our t_max
            )

            if t_max > optimal_t_max:
                optimal_t_max = t_max

    return optimal_t_max


def get_optimized_t_max_2(dataset: List[WindowData]) -> int:
    """
    Based off of the dataset contained in `dataset`, compute the lowest possible `t_max_2` value and return it.
    """
    optimal_t_max = 0

    # compute every 5 minutes (matches training loop)
    time_range = torch.arange(0, 1, 300/86400)
    num_timepoints = len(time_range)
    # assumes that all days have same number of trips (they should)
    num_trips = len(dataset[0].start_times)
    # expands into a matrix `num_timepoints` x `num_trips`
    time_range = time_range.repeat(num_trips, 1).t()

    for day in tqdm(dataset, desc="Computing optimal t_max value"):
        # duplicate start and end times by num_timepoints
        start_times = day.start_times.repeat(num_timepoints, 1)
        end_times = day.end_times.repeat(num_timepoints, 1)

        # now that dimensions match, compute all windowed times throughout the day
        start_times = (start_times <= time_range).int()
        end_times = (end_times >= time_range).int()

        # multiply together to compute windows
        combined_times = start_times * end_times

        # because of space constaints, need to iterate this
        for window in tqdm(combined_times, desc="Iterating by day", leave=False):
            # apply window mask
            node_visits = day.node_visits * window

            # type 2: simply sum node_visit items together, count nonzero
            node_visits = node_visits.sum(dim=0)

            t_max = node_visits.count_nonzero()

            if t_max > optimal_t_max:
                optimal_t_max = t_max

    return optimal_t_max


class ModelA(torch.nn.Module):
    """
    Window-based GNN model for real-time transit delay estimation. Type A (4 layers, large layers in between)
    """

    def __init__(self, t_max: int, n_t: int, args: argparse.Namespace):
        super().__init__()
        if args.no_trip_id and not args.type_2_window:
            node_features = t_max  # type 1, no trip
        elif args.no_trip_id:  # type 2, no trip
            node_features = t_max * 2
        else:  # type 2, incl. trip (and schedule)
            node_features = t_max * 3
        self.conv1 = GCNConv(node_features, n_t)
        self.conv2 = GCNConv(n_t, n_t*3)
        self.conv3 = GCNConv(n_t*3, n_t*2)
        self.conv4 = GCNConv(n_t*2, node_features)
        self.linear = Linear(node_features, t_max)

        self.args = args

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv4(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        x = self.linear(x)

        if not self.args.schedule_deviation:
            x = F.relu(x)

        return x


class Model(ModelA):
    """Compatibility class so that older models can be de-serialized"""
    pass


class ModelAS(torch.nn.Module):
    """
    Window-based GNN model for real-time transit delay estimation. Type AS (Type A with GraphSAGE)
    """

    def __init__(self, t_max: int, n_t: int, args: argparse.Namespace):
        super().__init__()
        if args.no_trip_id and not args.type_2_window:
            node_features = t_max  # type 1, no trip
        elif args.no_trip_id:  # type 2, no trip
            node_features = t_max * 2
        else:  # type 2, incl. trip (and schedule)
            node_features = t_max * 3
        self.conv1 = SAGEConv(node_features, n_t)
        self.conv2 = SAGEConv(n_t, n_t)
        self.conv3 = SAGEConv(n_t, n_t)
        self.conv4 = SAGEConv(n_t, node_features)
        self.linear = Linear(node_features, t_max)

        self.args = args

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv4(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        x = self.linear(x)

        if not self.args.schedule_deviation:
            x = F.relu(x)

        return x


class ModelB(torch.nn.Module):
    """
    Window-based GNN model for real-time transit delay estimation. Type B (8 layers, smaller layers in between)
    """

    def __init__(self, t_max: int, n_t: int, args: argparse.Namespace):
        super().__init__()
        if args.no_trip_id and not args.type_2_window:
            node_features = t_max  # type 1, no trip
        elif args.no_trip_id:  # type 2, no trip
            node_features = t_max * 2
        else:  # type 2, incl. trip (and schedule)
            node_features = t_max * 3

        self.conv_layers = []
        for i in range(8):
            if i == 0:
                self.conv_layers.append(GCNConv(node_features, n_t))
                continue
            if i == 7:
                self.conv_layers.append(GCNConv(n_t, node_features))
                continue

            self.conv_layers.append(GCNConv(n_t, n_t))

        self.linear1 = Linear(node_features, t_max)
        self.linear2 = Linear(t_max, t_max)

        self.args = args

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index

        for conv_layer in self.conv_layers:
            x = conv_layer(x, edge_index)
            x = F.leaky_relu(x)
            x = F.dropout(x, training=self.training)

        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)

        if not self.args.schedule_deviation:
            x = F.relu(x)

        return x


class ModelP(torch.nn.Module):
    """
    Window-based GNN model for real-time transit delay estimation. Type P (GCN and pooling layers!)
    """

    def __init__(self, t_max: int, n_t: int, args: argparse.Namespace):
        super().__init__()
        if args.no_trip_id and not args.type_2_window:
            node_features = t_max  # type 1, no trip
        elif args.no_trip_id:  # type 2, no trip
            node_features = t_max * 2
        else:  # type 2, incl. trip (and schedule)
            node_features = t_max * 3

        # linear to scale up to `n_t`, then two convolutions before pooling
        self.lin1 = Linear(node_features, n_t)
        self.conv1 = GCNConv(n_t, n_t)
        self.conv2 = GCNConv(n_t, n_t)

        # pooling
        self.pool = SAGPooling(n_t)

        # convolutions before linear
        self.conv3 = GCNConv(n_t, n_t)
        self.conv4 = GCNConv(n_t, n_t)
        self.conv5 = GCNConv(n_t, node_features)
        self.conv6 = GCNConv(node_features, t_max)

        # one linear before out
        self.lin2 = Linear(t_max, t_max)

        self.args = args

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index

        # linear, then 2 convolutions before pooling
        x = self.lin1(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)

        # pooling
        x, edge_index, _, _, _, _ = self.pool(x, edge_index)

        # other layers
        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv4(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv5(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv6(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)

        x = self.lin2(x)

        if not self.args.schedule_deviation:
            x = F.relu(x)

        return x


class ModelL(torch.nn.Module):
    """
    Window-based GNN model for real-time transit delay estimation. Type L (just linear)
    """

    def __init__(self, t_max: int, n_t: int, args: argparse.Namespace):
        super().__init__()
        if args.no_trip_id and not args.type_2_window:
            node_features = t_max  # type 1, no trip
        elif args.no_trip_id:  # type 2, no trip
            node_features = t_max * 2
        else:  # type 2, incl. trip (and schedule)
            node_features = t_max * 3

        self.lin1 = Linear(node_features, n_t)
        self.lin2 = Linear(n_t, n_t)
        self.lin3 = Linear(n_t, n_t)
        self.lin4 = Linear(n_t, t_max)

        self.args = args

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index

        # linear, then 2 convolutions before pooling
        x = self.lin1(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)

        x = self.lin2(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)

        x = self.lin3(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)

        x = self.lin4(x)

        if not self.args.schedule_deviation:
            x = F.relu(x)

        return x


class ModelS(torch.nn.Module):
    """
    Window-based GNN model for real-time transit delay estimation. Type S (simple, one graph layer and linear)
    """

    def __init__(self, t_max: int, n_t: int, args: argparse.Namespace):
        super().__init__()
        if args.no_trip_id and not args.type_2_window:
            node_features = t_max  # type 1, no trip
        elif args.no_trip_id and args.no_schedule:  # type 2, no trip and no schedule
            node_features = t_max  # probably not a great idea lol
        # type 2, no trip or no schedule (not both)
        elif args.no_trip_id or args.no_schedule:
            node_features = t_max * 2
        else:  # type 2, incl. trip (and schedule)
            node_features = t_max * 3

        self.lin1 = Linear(node_features, n_t)
        self.graph1 = GraphConv(n_t, n_t)
        self.graph2 = GraphConv(n_t, n_t)
        self.graph3 = GraphConv(n_t, n_t)
        self.graph4 = GraphConv(n_t, n_t)
        self.lin2 = Linear(n_t, t_max)

        self.args = args

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index

        x = self.lin1(x)
        x = F.tanh(x)

        x = self.graph1(x, edge_index)
        x = F.tanh(x)
        # x = F.dropout(x, training=self.training)

        x = self.graph2(x, edge_index)
        x = F.tanh(x)
        # x = F.dropout(x, training=self.training)

        x = self.graph3(x, edge_index)
        x = F.tanh(x)
        # x = F.dropout(x, training=self.training)

        x = self.graph4(x, edge_index)
        x = F.tanh(x)
        # x = F.dropout(x, training=self.training)

        x = self.lin2(x)

        if not self.args.schedule_deviation:
            x = F.relu(x)
        else:
            x = F.tanh(x)

        return x


class ModelSD(torch.nn.Module):
    """
    Window-based GNN model for real-time transit delay estimation. Type SD (simple, but deep)
    """

    def __init__(self, t_max: int, n_t: int, args: argparse.Namespace):
        super().__init__()
        if args.no_trip_id and not args.type_2_window:
            node_features = t_max  # type 1, no trip
        elif args.no_trip_id and args.no_schedule:  # type 2, no trip and no schedule
            node_features = t_max  # probably not a great idea lol
        # type 2, no trip or no schedule (not both)
        elif args.no_trip_id or args.no_schedule:
            node_features = t_max * 2
        else:  # type 2, incl. trip (and schedule)
            node_features = t_max * 3

        self.lin1 = Linear(node_features, node_features)
        self.graph1 = GraphConv(node_features, node_features)
        self.graph2 = GraphConv(node_features, node_features)
        self.graph3 = GraphConv(node_features, node_features)
        self.graph4 = GraphConv(node_features, node_features)
        self.graph5 = GraphConv(node_features, t_max)
        self.graph6 = GraphConv(t_max, t_max)
        self.graph7 = GraphConv(t_max, t_max)
        self.graph8 = GraphConv(t_max, t_max)
        self.graph9 = GraphConv(t_max, t_max)
        self.graph10 = GraphConv(t_max, t_max)
        self.lin2 = Linear(t_max, t_max)

        self.args = args

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index

        x = self.lin1(x)
        x = F.tanh(x)

        x = self.graph1(x, edge_index)
        x = F.tanh(x)
        # x = F.dropout(x, training=self.training)

        x = self.graph2(x, edge_index)
        x = F.tanh(x)
        # x = F.dropout(x, training=self.training)

        x = self.graph3(x, edge_index)
        x = F.tanh(x)
        # x = F.dropout(x, training=self.training)

        x = self.graph4(x, edge_index)
        x = F.tanh(x)
        # x = F.dropout(x, training=self.training)

        x = self.graph5(x, edge_index)
        x = F.tanh(x)
        x = self.graph6(x, edge_index)
        x = F.tanh(x)
        x = self.graph7(x, edge_index)
        x = F.tanh(x)
        x = self.graph8(x, edge_index)
        x = F.tanh(x)
        x = self.graph9(x, edge_index)
        x = F.tanh(x)
        x = self.graph10(x, edge_index)
        x = F.tanh(x)

        x = self.lin2(x)

        if not self.args.schedule_deviation:
            x = F.relu(x)
        else:
            x = F.tanh(x)

        return x


class MSENonZeroLoss(torch.nn.Module):
    """
    Compute Mean Squared Error over only the non-zero elements in the target output vector. This is used
    to give a more meaningful metric in cases where the output vector is sparse, and as such an MSE that averaged
    over these values would be unhelpfully low.
    """

    def __init__(self):
        super(MSENonZeroLoss, self).__init__()

    def forward(self, predicted, target):
        # not concerned with whether or not the model actually predicted zero correctly
        predicted_nonzero = predicted[target != 0]
        target_nonzero = target[target != 0]

        error = (predicted_nonzero - target_nonzero)**2
        loss = torch.sum(error) / len(target_nonzero)
        return loss


class MAENonZeroLoss(torch.nn.Module):
    """
    Compute Mean Absolute Error over only the non-zero elements in the target output vector. This is used
    to give a more meaningful metric in cases where the output vector is sparse, and as such an MAE that averaged
    over these values would be unhelpfully low. Also, absolute value is used so that all errors are positive.
    """

    def __init__(self):
        super(MAENonZeroLoss, self).__init__()

    def forward(self, predicted, target):
        # not concerned with whether or not the model actually predicted zero correctly
        predicted_nonzero = predicted[target != 0]
        target_nonzero = target[target != 0]

        error = (predicted_nonzero - target_nonzero).abs()
        loss = torch.sum(error) / len(target_nonzero)
        return loss


class BinaryLossRestrictor(torch.nn.Module):
    """
    Filter over a loss function, only reporting loss values on items that fit within a certain range. Binary restriction,
    meaning that loss values are either entirely counted (if in the filter), or completely skipped (if not). Values are filtered
    based off of whether or not the item at the *target* index matches the filter.
    """

    def __init__(self, criterion):
        super(BinaryLossRestrictor, self).__init__()

        self.criterion = criterion

    def forward(self, predicted, target, lb, ub):
        """
        Compute loss on predicted and target, only on values between `lb` and `ub` in target.
        """
        index = (target >= lb) & (target <= ub)
        predicted = predicted[index]
        target = target[index]

        return self.criterion(predicted, target)

    def forward(self, predicted, target, mask):
        """
        Compute loss on predicted and target, only on values that are matched by `mask`.
        """
        predicted = predicted[mask]
        target = target[mask]

        return self.criterion(predicted, target)


def nonzero_std(input: torch.Tensor, mask: Union[torch.Tensor, None] = None, dim: int = 0):
    """
    Compute standard deviation, while ignoring zero values. This can be used (and is used by UniformityPenalty) to
    compute a more meaningful standard deviation on sparse data.

    Optional parameter `mask` can be used to specify which values in input should be treated as non-zero, instead
    of what input as as nonzero.
    """
    # return torch.sqrt(
    #     torch.nanmean(
    #         torch.pow(
    #             torch.abs(
    #                 input - torch.nanmean(input, dim=dim).unsqueeze(dim)
    #             ), 2
    #         ), dim=dim
    #     )
    # )

    # means = torch.nanmean(input, dim=dim)
    # means = means.nan_to_num(0.0)
    # residuals = input - means.unsqueeze(dim)
    # residuals = torch.pow(residuals, 2)
    # return torch.sqrt(torch.nanmean(residuals, dim=dim))

    if mask is None:
        mask = input

    # compute mean over non-zero elements manually
    sums = input.masked_fill(mask == 0, 0).sum(dim=dim)  # only sum over non-zero elements
    # get non-zero counts per column, by converting to bool (so zero values are false), then back to int (true = 1)
    nonzero_counts = mask.bool().int().sum(dim=dim)
    means = sums / nonzero_counts
    means = means.nan_to_num(nan=0, posinf=0, neginf=0)  # for empty values, this propagates a zero instead

    # compute residuals, set ones with mask as zero to zero (not counting these) 
    residuals = input - means.unsqueeze(dim)
    residuals[residuals == 0] = 0.00000001  # use small epsilon value to note that the residual here should still be counted
    residuals[mask == 0] = 0
    residuals = torch.pow(residuals, 2)

    # compute standard deviation of non-zero residuals
    std = residuals.sum(dim=dim) / residuals.bool().int().sum(dim=dim)
    std = std.nan_to_num(nan=0, posinf=0, neginf=0)  # once again, all zero residuals treated as std zero
    return torch.sqrt(std)


class UniformityPenalty(torch.nn.Module):
    """
    Compute a penalty value for when feature distribution across vectors is less diverse in the
    predicted values compared to the target.

    Specifically, this computes the average difference between the column-wise standard deviations of
    the target tensor, compared to the predicted tensor. Standard deviations greater than the target tensor
    are skipped.
    """

    def __init__(self):
        super(UniformityPenalty, self).__init__()

    def forward(self, predicted, target):
        pred_std = nonzero_std(predicted, mask=target, dim=0)
        target_std = nonzero_std(target, dim=0)
        penalties = target_std - pred_std

        penalties[penalties < 0] = 0
        if len(penalties[penalties > 0]) == 0:
            return 0  # this avoids nan on divide by zero here

        # just returning the sum of penalties now
        penalty_val = torch.sum(penalties)  # / len(penalties[penalties > 0])
        return penalty_val


def get_parser() -> argparse.ArgumentParser:
    """
    Return an argument parser to be used in training scripts for this model.
    """
    parser = argparse.ArgumentParser(
        description="Script used to train the window_gnn model"
    )

    params = [
        ("lr", 0.001, "Learning rate"),
        ("batch-size", 32, "Batch size"),
        ("epochs", 25, "Number of epochs"),
        # if trip ID vector is added back, this will be helpful again
        ("no-trip-id", False, "Disable trip ID vector in input vector"),
        ("no-schedule", False, "Remove schedule vector in input vector"),
        ("zero-in-loss", False,
         "Include zero values in loss calculations when training model"),
        ("shuffle", False,
         "Shuffle data before each epoch"),
        ("type-2-window", False,
         "Enable type 2 window format (trips share same index across nodes)"),
        ("only-pred-schedule", False,
         "Limit schedule information to visits we are trying to predict"),
        ("schedule-deviation", False,
         "Train the model to predict schedule deviation values (including negative values) rather than absolute times"),
        ("lr-scheduler", 0.0,
         "Enable an exponential learning rate scheduler with specified decay rate"),
        ("lr-scheduler-cyclic", 0.0,
         "Enable a cyclic learning rate scheduler, with max_lr specified here, base_lr as lr parameter"),
        ("mse", False,
         "Use MSE instead of MAE for training"),
        ("test-model", "", "Test provided model instead of training a new one (do not include file extension in model name)"),
        ("test-cuda", False, "Test using CUDA instead of CPU"),
        ("model", 'A', "Model to train"),
        ("normalize", False, "Normalize input features"),
        ("early-stopping", 0,
         "Enable early stopping with specified patience parameter, in epochs"),
        ("forward-limit", 0, "Limit predictions to specified number of minutes past current time in a given window"),
        ("weight-decay", 0.0,
         "Apply weight decay parameter to loss function in optimizer (L2 regularization)"),
        ("outlier-cut", 0.0,
         "Don't train on values with abs(deviations) >= outlier-cut (in minutes)"),
        ("uniformity-penalty", 0.0,
         "Impose a loss penalty of specified multiplier for features that are more uniform than ground truth"),
    ]  # (varname, default value, description) pairs

    for (param, default, desc) in params:
        action = "store_true" if (type(default) == bool) else "store"

        if action == "store":
            parser.add_argument(f"--{param}", default=default, help=f"{desc}, default: {default}",
                                type=type(default), action=action)
        else:
            parser.add_argument(f"--{param}", default=default,
                                help=f"{desc}, default: {default}", action=action)

    return parser


MODEL_MAP = {'A': ModelA, 'B': ModelB, 'AS': ModelAS,
             'P': ModelP, 'L': ModelL, 'S': ModelS, 'SD': ModelSD}


def training_loop(data: Iterator[WindowData], t_max: int, n_t: int, args: argparse.Namespace, logfile: str, cpu_cache: bool = False) -> ModelA:
    """
    Train the model for `num_epochs` epochs, on the data contained in `data`. The `data` iterator
    should separate WindowData objects by day. Also, make sure every data point in `data` has the same `t_max` value :)

    `args` should be passed from `parser.get_args()` to parse in command line arguments for hyperparameters. `logfile` is the name
    of the file logging should be done to. `cpu_cache` indicates whether or not to use a WindowCacheDataset.
    """
    file = open(logfile, 'w')
    file.write(f"Args for this training session: {vars(args)}\n")

    if args.model == "GCN":
        model = GCN(t_max * 3, t_max, 20, t_max, dropout=0.3, act='tanh')
    else:
        model = MODEL_MAP[args.model](t_max, n_t, args)

    if args.zero_in_loss:
        criterion = L1Loss() if not args.mse else MSELoss()
    else:
        criterion = MAENonZeroLoss() if not args.mse else MSENonZeroLoss()

    if args.forward_limit:
        criterion = BinaryLossRestrictor(criterion)

    if args.uniformity_penalty:
        penalty_criterion = UniformityPenalty()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.outlier_cut == 0.0:
        args.outlier_cut = None
    else:
        args.outlier_cut *= (60/86400)  # convert to time range

    model.train()
    if not cpu_cache:
        data = WindowDataset(data, torch.arange(
            0, 1, 300/86400), type_2=args.type_2_window, only_pred_schedule=args.only_pred_schedule, schedule_deviation=args.schedule_deviation, normalize=args.normalize, outlier_cut=args.outlier_cut)
    else:
        data = WindowCacheDataset(data, torch.arange(
            0, 1, 300/86400), type_2=args.type_2_window, only_pred_schedule=args.only_pred_schedule)

    # early stopping parameters
    best_val_loss = float('inf')
    best_params = None
    patience_left = args.early_stopping

    if not args.early_stopping:
        # no early stopping, training set is just the whole dataset
        training_set = data
    else:
        # create train and validation split
        data.shuffle()
        splitting_point = int(len(data) * 0.8)
        training_set = data[:splitting_point]
        validation_set = data[splitting_point:]

    if args.lr_scheduler:
        scheduler = ExponentialLR(optimizer, gamma=args.lr_scheduler)
    elif args.lr_scheduler_cyclic:
        scheduler = CyclicLR(optimizer, base_lr=args.lr, max_lr=args.lr_scheduler_cyclic,
                             step_size_up=4*(len(training_set)/args.batch_size), cycle_momentum=False)

    for epoch in (pbar := tqdm(range(args.epochs), desc="Training model, epoch 0")):
        cumulative_loss_val = torch.tensor(0.0)
        steps_into_batch = 0

        if args.shuffle:
            training_set.shuffle()

        for i, windowed_data in tqdm(enumerate(training_set), desc="Iterating across dataset", leave=False, total=len(training_set)):
            optimizer.zero_grad()
            if args.no_trip_id:
                # remove trip ID portion of the input vector
                windowed_data.x = windowed_data.x[:, t_max:]
            if args.no_schedule and not args.no_trip_id:
                # remove schedule portion of input vector
                windowed_data.x = windowed_data.x[:, :2*t_max]
            if args.no_schedule and args.no_trip_id:
                # only send in the known time values
                windowed_data.x = windowed_data.x[:, t_max:2*t_max]

            if args.model == 'GCN':
                predicted_output = model(windowed_data.x, windowed_data.edge_index)
            else:
                predicted_output = model(windowed_data)
            target_output = windowed_data.y

            if target_output.sum() == 0:
                # unhelpful to train on these scenarios because the model doesn't need to make any predictions - skip!
                continue

            if args.forward_limit:
                # mask out values outside of the time range
                incr = (args.forward_limit * 60) / 86400
                mask = target_output.clone()

                if args.schedule_deviation:
                    # add the schedule back to mask to get absolute times
                    mask += training_set[i].x[:, 2*t_max:]
                    mask[target_output == 0] = 0  # make sure these stay zero

                t_val = training_set.idx_map[i][1]
                t_val_max = t_val + incr
                mask = (mask >= t_val) & (mask <= t_val_max)

                loss = criterion(predicted_output, target_output, mask)
            else:
                loss = criterion(predicted_output, target_output)

            if args.uniformity_penalty:
                loss += args.uniformity_penalty * \
                    penalty_criterion(predicted_output, target_output)

            cumulative_loss_val += loss
            steps_into_batch += 1

            if steps_into_batch == args.batch_size or i == len(training_set) - 1:
                batch_loss = cumulative_loss_val / args.batch_size
                batch_loss.backward()
                optimizer.step()

                cumulative_loss_val = torch.tensor(0.0)
                steps_into_batch = 0

                if args.lr_scheduler_cyclic:
                    scheduler.step()

        if args.early_stopping:
            # test on validation set, update params, and break if loss hasn't improved in patience epochs
            _, mae = test(model, validation_set, args)
            model.train()
            pbar.set_description(
                f"Training model, epoch {epoch+1}, loss={loss.item()}, val_loss={mae}")
            file.write(
                f"Loss for epoch {epoch}: {loss.item()}, val_loss={mae}\n")

            if mae < best_val_loss:
                # nice! update model params
                best_params = copy.deepcopy(model.state_dict())
                patience_left = args.early_stopping
                best_val_loss = mae
            else:
                # not an improvement :( see how much patience is left
                patience_left -= 1
                if patience_left == 0:
                    # early stopping indeed
                    break
        else:
            pbar.set_description(
                f"Training model, epoch {epoch+1}, loss={loss.item()}")
            file.write(f"Loss for epoch {epoch}: {loss.item()}\n")

        if args.lr_scheduler:
            scheduler.step()

    file.close()
    model.load_state_dict(best_params)  # want the best model to be saved
    return model


def test(model: ModelA, data: Union[Iterator[WindowData], WindowDataset], args: argparse.Namespace) -> float:
    """
    Test the model on provided data, returning a final average (MSE, MAE) tuple (using the custom MSENonZero/MAENonZero loss functions).
    """
    model.eval()

    mse = MSENonZeroLoss()
    mae = MAENonZeroLoss()
    mse_val = 0
    mae_val = 0

    if args.forward_limit:
        mse = BinaryLossRestrictor(mse)
        mae = BinaryLossRestrictor(mae)

    if type(data) != WindowDataset:
        t_max = data[0].t_max if not args.type_2_window else data[0].t_max_2
        data = WindowDataset(data, torch.arange(
            0, 1, 300/86400), type_2=args.type_2_window, only_pred_schedule=args.only_pred_schedule, schedule_deviation=args.schedule_deviation, normalize=args.normalize, outlier_cut=args.outlier_cut)
    else:
        t_max = data.data[0].t_max if not args.type_2_window else data.data[0].t_max_2

    count = 0
    for i, windowed_data in tqdm(enumerate(data), desc="Iterating across dataset (test loop)", leave=False, total=len(data)):
        if args.no_trip_id:
            # remove trip ID half of the input vector
            windowed_data.x = windowed_data.x[:, t_max:]
        if args.no_schedule and not args.no_trip_id:
            # remove schedule portion of input vector
            windowed_data.x = windowed_data.x[:, :2*t_max]
        if args.no_schedule and args.no_trip_id:
            # only send in the known time values
            windowed_data.x = windowed_data.x[:, t_max:2*t_max]

        if args.model == 'GCN':
            predicted_output = model(windowed_data.x, windowed_data.edge_index)
        else:
            predicted_output = model(windowed_data)
        target_output = windowed_data.y

        if target_output.sum() == 0:
            # unhelpful to test on these scenarios because the model doesn't need to make any predictions - skip!
            continue

        if args.schedule_deviation and args.normalize:
            # want to de-normalize these values so that the loss values in testing are meaningful
            selector = predicted_output[target_output != 0]
            predicted_output[target_output != 0] = (
                selector * 0.010416) - 0.000347

            selector = target_output[target_output != 0]
            target_output[target_output != 0] = (
                selector * 0.010416) - 0.000347

        if args.forward_limit:
            # mask out values outside of the time range
            incr = (args.forward_limit * 60) / 86400
            mask = target_output.clone()

            if args.schedule_deviation:
                # add the schedule back to mask to get absolute times
                mask += data[i].x[:, 2*t_max:]
                mask[target_output == 0] = 0  # make sure these stay zero

            t_val = data.idx_map[i][1]
            t_val_max = t_val + incr
            mask = (mask >= t_val) & (mask <= t_val_max)

            loss_mae = mae(predicted_output, target_output, mask)
            loss_mse = mse(predicted_output, target_output, mask)
        else:
            loss_mae = mae(predicted_output, target_output)
            loss_mse = mse(predicted_output, target_output)

        if math.isnan(loss_mae) or math.isnan(loss_mse):
            continue

        count += 1
        prev = max(1, count-1)
        loss = loss_mse
        mse_val += loss.item() / prev
        mse_val *= prev/count

        loss = loss_mae
        mae_val += loss.item() / prev
        mae_val *= prev/count

    return (mse_val, mae_val)
