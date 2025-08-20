"""
simple_gnn: First iteration of a GNN model for real-time vehicle arrival prediction.

This model operates on graphs of interconnected stop nodes whose only feature is a set of (trip ID, time) pairings.

Author: Nick Gable (gable105@umn.edu)
"""

import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, Linear
import torch.nn.functional as F
from realtime_transit.graphs import TransitNetwork, TransitStop, TransitEdge
from typing import Iterator
from tqdm import tqdm
import math

# allows for 50 visits to a stop per vehicle - TODO: Actually calculate this from MT dataset
NODE_FEATURES = 1500


def to_data(network: TransitNetwork, node_features=NODE_FEATURES) -> Data:
    """
    Convert TransitNetwork high level information into the graph representation needed
    for this model. Optionally, make a different sized model than the constant by changing
    the `node_features` parameter.

    This function handles NaT values by giving a seconds value of 0.
    """
    edges = []  # list of (source node, dest node) tuples

    # list of node features: feature format is alternating (trip index, seconds since 4 AM) pairs
    nodes = []

    trip_map = {trip_id: idx for (idx, trip_id) in enumerate(
        network.trips)}  # key: trip_id, value: trip index

    # "count out" stops so that they are zero indexed
    stop_indices = {}  # key: stop_id, value: graph index (0 indexed)
    current_idx = 0
    for stop_id in network.stops.keys():
        stop_indices[stop_id] = current_idx
        current_idx += 1

        # initialize node values
        nodes.append(node_features * [0])

    # enumerate through stops, generating edge and node lists
    for stop_id, stop in tqdm(network.stops.items(), desc="Populating edge list"):
        stop: TransitStop = stop

        # edge enumeration
        for edge in stop.outgoing_edges:
            edges.append((stop_indices[stop.stop_id],
                         stop_indices[edge]))

        # node enumeration
        for trip_id, time in stop.visits.items():
            (hour, minute) = time
            if math.isnan(hour) or math.isnan(minute):
                seconds = 0  # no crossing recorded, but should have had a crossing
            else:
                hour = hour if hour > 3 else hour + 12  # move first three hours to end of day
                seconds = 3600*(hour-4) + 60*minute

            stop_idx = stop_indices[stop_id]

            # find open slot in this node, and insert there
            for i in range(0, len(nodes[stop_idx]), 2):
                if nodes[stop_idx][i] == 0:
                    nodes[stop_idx][i] = trip_map[trip_id]
                    nodes[stop_idx][i+1] = seconds
                    break

    # build Data object and return
    edge_index = torch.tensor(edges, dtype=torch.int64)
    x = torch.tensor(nodes, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index.t().contiguous())


def to_labeled_data(data: Data, cutoff: int, show_progress: bool = True) -> Data:
    """
    Given data `data` (from function `to_data`) and cutoff time `cutoff` (in seconds),
    transform `data` so that the `x` input feature array masks any stop crossings reported after
    the cutoff time, instead moving those to the `y` output feature array, to be predicted by the model.

    `show_progress` controls whether or not tqdm progress bar logging is used

    Note: Currently `data.y = data.x`, meaning that the model will output the existing known values in addition
    to the new ones.
    """
    result = data.clone()
    result.y = result.x.clone()

    for i in tqdm(range(len(result.x)), desc="Masking x feature matrix", disable=(not show_progress)):
        for j in range(0, len(result.x[i]), 2):
            if result.x[i][j+1] >= cutoff:
                # mask from `x`
                result.x[i][j] = -1
                result.x[i][j+1] = -1

    return result


def generate_mask(data: Data, cutoff: int, show_progress: bool = True) -> torch.Tensor:
    """
    Given data `data` (from function `to_data`) and cutoff time `cutoff` (in seconds),
    return an input mask tensor of the same dimensions as data, that masks out any input features
    after the cutoff time.
    """
    result = torch.ones(size=data.x.size())
    for i in tqdm(range(len(data.x)), desc="Creating input mask", disable=(not show_progress)):
        for j in range(0, len(data.x[i]), 2):
            if data.x[i][j+1] >= cutoff:
                # mask remaining values in this row, starting at j
                result[i][j:] = 0
                break  # no need to iterate further

    return result


class Model(torch.nn.Module):
    """
    Simple GNN model for real-time transit delay estimation.
    """

    def __init__(self, node_features: int = NODE_FEATURES):
        super().__init__()
        self.conv1 = GCNConv(node_features, int(node_features * 5))
        self.conv2 = GCNConv(int(node_features * 5), node_features)

    def forward(self, data: Data, input_mask: torch.Tensor):
        x, edge_index = data.x, data.edge_index

        x = x * input_mask

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x
    
class LinearModel(torch.nn.Module):
    """
    Simple GNN model for real-time transit delay estimation. Testing model uses a single linear layer.
    """

    def __init__(self, node_features: int = NODE_FEATURES):
        super().__init__()
        self.conv1 = Linear(node_features, node_features)
        # self.conv2 = GCNConv(int(node_features * 5), node_features)

    def forward(self, data: Data, input_mask: torch.Tensor):
        x, edge_index = data.x, data.edge_index

        x = x * input_mask

        x = self.conv1(x)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x = self.conv2(x, edge_index)

        return x


def training_loop(num_epochs: int, data: Iterator[Data]) -> Model:
    """
    Train the model for `num_epochs` epochs, on the data contained in `data`. The `data` iterator
    should separate Data objects by day.
    """
    model = Model(len(data[0].x[0]))

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in (pbar := tqdm(range(num_epochs), desc="Training model, epoch 1")):
        for day in tqdm(data, desc="Iterating across days", leave=False):
            # generate mask matrix where each element maps to `x`, giving the time value
            # that element in `x` should be masked at
            mask_matrix = torch.clone(day.x)
            
            # this fancy command replaces every even column with the odd column next to it 
            mask_matrix.t()[::2] = day.x.t()[1::2]
            
            optimizer.zero_grad()

            # progressively add more data into prediction
            for time in tqdm(range(0, 86400, 300), desc="Iterating within day", leave=False):
                # labeled = to_labeled_data(day, time, show_progress=False)
                #input_mask = generate_mask(day, cutoff=time, show_progress=False)
                
                # quick input mask is simply a boolean check if the mask matrix values are in the time window
                input_mask = (mask_matrix >= time).float()

                predicted_output = model(day, input_mask)
                target_output = day.y

                loss = criterion(predicted_output, target_output)

                loss.backward()
                optimizer.step()

        pbar.set_description(
            f"Training model, epoch {epoch+1}, loss={loss.item()}")

    return model
