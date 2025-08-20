"""
Abstract representations of transit network graphs. Used as a high level interface to examine graph-wide network information,
with conversion options to and from formats used with modelling.

TODO: Map rendering process is pretty inefficient, would be helpful to improve for live demos (caching helpful).

Author: Nick Gable (gable105@umn.edu)
"""


from typing import Tuple, Dict, Set, List
import folium
import pandas as pd
from tqdm import tqdm


class TransitNetwork:
    """
    Abstract representation of transit network graph. Implemented as a directed graph.
    """

    def __init__(self, hour: int = 4, minute: int = 0):
        """
        Create a new TransitNetwork, optionally setting the current time in the network.
        """
        self.stops: Dict[int, TransitStop] = {
        }  # key: stop_id, val: TransitStop
        # all trips that are active on this network
        self.trips: Set[str] = set()
        self.hour = hour
        self.minute = minute

    def consolidate_trips(self):
        """
        Consolidate trips, converting self.trips into a list instead of a set. Call this after all trips have 
        been processed, and before converting this network into a lower level representation.
        """
        self.trips: List[str] = list(self.trips)

    def reset_trips(self):
        """
        Clear stored trips, converting self.trips back into a set and erasing all recorded trips. Call this to re-use
        the network on a different dataset. 

        Also resets stored visits at each stop in the network.
        """
        self.trips: Set[str] = set()

        for stop in self.stops.values():
            stop.visits = {}

    def connect(self, stop_a: int, stop_b: int, route: int):
        """
        Connect two transit stops to each other on the graph. `stop_a` and `stop_b` are stop_id values. Does nothing
        if these two stops are already connected. Also, updates stops with route information.
        """
        stop_a: TransitStop = self.stops[stop_a]
        stop_b: TransitStop = self.stops[stop_b]

        edge = TransitEdge(stop_a, stop_b, self)
        edge.add_route(route)
        stop_a.add_route(route)
        stop_b.add_route(route)

        # only connect if there isn't already a connection
        if stop_a.stop_id not in stop_b.incoming_edges.keys():
            stop_a.outgoing_edges[stop_b.stop_id] = edge
            stop_b.incoming_edges[stop_a.stop_id] = edge
        else:
            # only update on stop a, because it is the same edge object on stop B side
            stop_a.outgoing_edges[stop_b.stop_id].add_route(route)

    def get_map(self):
        """
        Returns Folium map encoded with network information.
        """
        first_coords = list(self.stops.values())[0].get_coords()
        map = folium.Map(location=first_coords,
                         zoom_start=16, control_scale=True)
        edge_count = 0
        for stop in tqdm(self.stops.values(), desc="Adding stops to map"):
            stop: TransitStop = stop
            folium.CircleMarker(stop.get_coords(
            ), radius=5, tooltip=str(stop), fill=True, color="blue").add_to(map)

            # draw all outgoing edges from this stop

            for edge in stop.outgoing_edges.values():
                edge_count += 1
                edge: TransitEdge = edge
                stop_a_coords = edge.stop_a.get_coords()
                stop_b_coords = edge.stop_b.get_coords()
                folium.PolyLine([stop_a_coords, stop_b_coords],
                                tooltip=str(edge),
                                color="purple").add_to(map)

        print(f"Added {edge_count} edges")

        return map

    def register_visits(self, visit_data: pd.DataFrame):
        """
        Taking in stop crossing data `visit_data`, register visits at appropriate stops in this network.

        Note: Visits recorded in visit_data without a timestamp (i.e. visits that should have happened but
        were not conclusively recorded using location data) propagate as (nan, nan) visits here. Model conversion
        functions should be aware of this and adjust accordingly to avoid problems.
        """
        visit_data.date = pd.to_datetime(
            visit_data.date)  # convert to datetime for our use
        for visit in tqdm(visit_data.itertuples(), desc="Registering visits", total=len(visit_data)):
            scheduled_hour = int(visit.arrival_time.split(":")[0])
            scheduled_minute = int(visit.arrival_time.split(":")[1])

            if scheduled_hour >= 24:
                # sometimes this is done: need to move to the start of the day to be consistent
                scheduled_hour -= 24
            
            stop = self.stops[visit.stop_id]
            stop.register_visit(
                visit.trip_id, visit.date.hour, visit.date.minute, scheduled_hour, scheduled_minute)

    def __repr__(self):
        return f"TransitNetwork: {len(self.stops)} stops, {len(self.trips)} trips, time={(self.hour, self.minute)}"


class TransitStop:
    """
    Abstract representation of a stop in the transit network.
    """

    def __init__(self, stop_id: int, stop_lat: float, stop_lon: float, stop_name: str, outgoing_edges=None, incoming_edges=None, routes=None, network=None):
        self.stop_id = stop_id
        self.stop_lat = stop_lat
        self.stop_lon = stop_lon
        self.stop_name = stop_name
        self.visits = {}  # key: trip_id, value: (hour, minute, scheduled_hour, scheduled_minute)

        if not outgoing_edges:
            outgoing_edges = dict()
        if not incoming_edges:
            incoming_edges = dict()
        if not routes:
            routes = list()

        self.outgoing_edges = outgoing_edges  # key: stop_id, val: TransitEdge
        self.incoming_edges = incoming_edges
        self.routes = routes
        self.network = network

    def get_coords(self) -> Tuple[float, float]:
        return (self.stop_lat, self.stop_lon)

    def add_route(self, route: int):
        if route not in self.routes:
            self.routes.append(route)

    def register_visit(self, trip_id: str, hour: int, minute: int, scheduled_hour: int, scheduled_minute: int):
        """Register a visit (predicted or past) for this vehicle at this stop."""
        self.visits[trip_id] = (hour, minute, scheduled_hour, scheduled_minute)
        self.network.trips.add(trip_id)

    def __str__(self):
        result = f"<b>{self.stop_name}</b><br /><b>ID: </b>{self.stop_id}<br /><b>Routes: </b>{','.join([str(i) for i in self.routes])}"

        for (trip_id, time) in self.visits.items():
            result += f"<br /><b>{trip_id}</b>: {time[0]}:{time[1]}, sch: {time[2]}:{time[3]}"

        return result

    def __repr__(self):
        result = f"{self.stop_name}: id={self.stop_id}, routes={self.routes}, num_visits={len(self.visits)}"
        return result


class TransitEdge:
    """
    Abstract representation of a connection between two stops in the transit network.
    """

    def __init__(self, stop_a: TransitStop, stop_b: TransitStop, network: TransitNetwork = None, routes=None):
        self.stop_a = stop_a
        self.stop_b = stop_b
        self.network = network
        if not routes:
            routes = list()
        self.routes = routes

    def add_route(self, route: int):
        if route not in self.routes:
            self.routes.append(route)

    def __str__(self):
        return f"<b>{self.stop_a.stop_name}</b> --> <b>{self.stop_b.stop_name}</b><br /><b>Routes: </b>{','.join([str(i) for i in self.routes])}"


def create_network(gtfs_stops: pd.DataFrame,
                   gtfs_stop_times: pd.DataFrame,
                   gtfs_trips: pd.DataFrame) -> TransitNetwork:
    """
    Construct a TransitNetwork based off of static GTFS data. This function does not load in stop crossing times, which
    is done separately with live gtfs_realtime data. 

    The three parameters for this function allow for control over which GTFS file / portions of a GTFS file are used. 
    Using the `gtfs_file` function from `util` will be helpful here.
    """
    network = TransitNetwork()

    # assemble stops
    for item in tqdm(gtfs_stops.itertuples(), desc="Assembling stops", total=len(gtfs_stops)):
        stop = TransitStop(item.stop_id, item.stop_lat,
                           item.stop_lon, item.stop_name, network=network)
        network.stops[stop.stop_id] = stop

    # now link by stop times (this process assumes gtfs is sorted by trip_id, then stop_sequence)
    # filter out trips that have the same service ID to speed things up (doesn't work, also used to get route info)
    gtfs_stop_times = gtfs_stop_times.merge(
        gtfs_trips, how="inner", on="trip_id")

    current_trip_id = None
    previous_stop_id = None
    for item in tqdm(gtfs_stop_times.itertuples(), desc="Connecting stops", total=len(gtfs_stop_times)):
        if item.trip_id == current_trip_id:
            # generate new edge
            network.connect(previous_stop_id, item.stop_id, item.route_id)
        else:
            # start of new trip id
            current_trip_id = item.trip_id
        previous_stop_id = item.stop_id

    return network
