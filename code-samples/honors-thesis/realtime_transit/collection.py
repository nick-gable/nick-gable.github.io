"""
collection: module used to automate data collection from GTFS-realtime feeds.

These functions create new processes that perform the actual collection process. As a result,
they can be called in an intepreter and closed as desired. However, it may be beneficial to keep
the shell open in order to see debugging output as it comes in.

Author: Nick Gable (gable105@umn.edu)
"""
from datetime import datetime, timedelta
from multiprocessing import Process
from time import sleep
import timeit

from realtime_transit.processing import get_realtime_locations, get_trip_updates

import pandas as pd
import os
import logging

WRITE_OUT_INTERVAL = 30  # write out to file every X mins

logger = logging.getLogger("collection")
# change this to adjust desired logging level
logging.basicConfig(level=logging.INFO)


def generate_file_names(prefix: str, filetype: str, begin: datetime, end: datetime) -> list[str]:
    """
    Returns a list of (start, end, file_name) tuples separated by transit day of operation, beginning with the date/time
    in `begin` and ending with the date/time in `end`. Prefixes these files with `prefix`, and gives them file
    extension `filetype`. start and end are `datetime` objects representing start of file (inclusive) and end of file
    (exclusive).

    This is a helper function for the collection functions, which call this function to split collection intervals
    into distinct files that are separated by transit day (4 AM - 4 AM).

    File format: `prefix`-YYYY-MM-DD-hhmmss-to-hhmmss.hd5
    """
    file_names = []
    file_start_date = begin
    while file_start_date < end:
        # if current date is earlier than 4 AM, and ending date is either not today, or today at or after 4 AM,
        # then end this file at 4 AM
        if file_start_date.hour < 4 and ((end.date() != file_start_date.date()) or end.hour >= 4):
            file_end_date = datetime(
                file_start_date.year, file_start_date.month, file_start_date.day, 4, 0, 0)
        # still have another day to go, set file end date to next day at 4 AM
        elif file_start_date.date() < end.date():
            file_end_date = datetime(
                file_start_date.year, file_start_date.month, file_start_date.day, 4, 0, 0)
            file_end_date += timedelta(days=1)

            # this check added for cases where final day is between 0-4 AM
            if file_end_date > end:
                file_end_date = end
        # last file: file end date is simply end parameter
        else:
            file_end_date = end

        file_names.append((file_start_date, file_end_date,
                           f"{prefix}-{file_start_date.strftime('%Y-%m-%d-%H%M%S')}-to-{file_end_date.strftime('%H%M%S')}.{filetype}"
                           ))
        file_start_date = file_end_date

    return file_names


def collection_process(begin: datetime, end: datetime, output_dir: str, collection_func, label: str, delay: int):
    """
    Helper function which does the bulk of the collection work for functions `realtime_process` and 
    `prediction_process`. It works by collecting data using the specified collection function (assumed to return
    a Pandas DataFrame), accumulating that data and splitting it into files based off of the provided `begin` and 
    `end` datetimes. Output files into `output_dir`. `delay` is the integer delay between collections. `label` is a 
    string description of the data collection used for file names and log messages.
    """
    logger.info(
        f"Starting {label} collection, waiting until start time")
    while datetime.now() < begin:
        # wait until current time is equal to begin time
        sleep((begin-datetime.now()).total_seconds())

    splits = generate_file_names(label, "csv.zip", begin, end)

    last_write = datetime.now()
    write_time = None  # use this to reduce delay after writes for more consistent collection
    for (_, finish, filename) in splits:
        try:
            logger.info(f"Starting file name {filename}")
            try:
                file_data = collection_func()
            except Exception as e:
                logger.error(
                    f"{type(e).__name__}: {str(e)} during initial {label} poll")
                logger.error(f"FATAL: ending {label} loop")
                break
            while datetime.now() < finish:
                try:
                    # wait before updating again, adjusting for write time delay
                    if write_time:
                        sleep(max(0, delay - write_time))
                        write_time = None
                    else:
                        sleep(delay)
                    
                    if datetime.now() > finish:
                        # we overslept!
                        break

                    logger.debug(
                        f"Retrieving and appending data for file {filename}")
                    try:
                        file_data = pd.concat(
                            [file_data, collection_func()])
                    except Exception as e:
                        logger.warning(
                            f"{type(e).__name__}: {str(e)} while attempting to poll {label}")
                        logger.warning(
                            f"Location data this cycle lost, trying again in {delay}s")
                        continue

                    if (datetime.now() - last_write) > timedelta(minutes=WRITE_OUT_INTERVAL):
                        # write out file
                        logger.info(
                            f"Incremental write out for file {filename}")
                        
                        now = timeit.default_timer()
                        file_data.to_csv(os.path.join(
                            output_dir, filename), index=False)
                        write_time = timeit.default_timer() - now

                        last_write = datetime.now()
                except Exception as e:
                    # error occured during main loop - will skip this iteration
                    logger.warning(
                        f"{type(e).__name__}: {str(e)} during {label} inner loop execution")
                    logger.warning(
                        f"{label} data this cycle lost, trying again in 15s")
        except KeyboardInterrupt:
            logger.info(
                f"Received keyboard interrupt, writing out current file and exiting ({label})")
            file_data.to_csv(os.path.join(output_dir, filename), index=False)
            break

        # write out final changes to file
        logger.info(f"Done with file {filename}, writing out")
        file_data.to_csv(os.path.join(output_dir, filename), index=False)

    logger.info(f"Done with {label} collection, process ending")


def realtime_process(begin: datetime, end: datetime, output_dir: str):
    """
    Perform realtime location data collection. This function should not be called directly unless
    you intend to have collection performed on the main thread as opposed to in a separate process - 
    otherwise, use `collect_realtime_locations` (see documentation there for more information).
    """
    collection_process(begin, end, output_dir,
                       get_realtime_locations, "rt-pos", 15)


def prediction_process(begin: datetime, end: datetime, output_dir: str, delay: int = 60):
    """
    Perform realtime prediction data collection. Similar to `realtime_process`, this function will
    block the current thread, and so it should not be called directly unless that is your intent. 
    """
    collection_process(begin, end, output_dir, get_trip_updates, "pred", delay)


def collect_realtime_locations(begin: datetime, end: datetime, output_dir: str) -> Process:
    """
    Collect realtime location data from Metro Transit GTFS-realtime feed, storing it in
    `csv.zip` files in `output_dir`.

    Parameters:
    - `begin`: `datetime` to begin collection at. Use `datetime.now()` if immediately is desired
    - `end`: `datetime` to end collection at; no location data will be saved past this point.
    - `output_dir`: directory to output files into

    Output file format: `rt-pos-YYYY-MM-DD-hhmmss-to-hhmmss.hd5`.

    New files will be created each time the function is called, or when a new transit day begins,
    which for Metro Transit is at 4 AM. 

    Returns new Process object representing the created and started process.
    """
    p = Process(target=realtime_process, args=(
        begin, end, output_dir), daemon=True)
    p.start()

    return p


def collect_predictions(begin: datetime, end: datetime, output_dir: str, delay: int = 60):
    """
    Collect realtime departure predictions from Metro Transit GTFS-realtime feed, storing it in
    `csv.zip` files in `output_dir`.

    See `collect_realtime_locations` documentation for details on parameters and technique. In this function,
    output files are prefixed with `pred`, and `delay` is an optional parameter used to change the polling delay.

    Returns new Process object representing the created and started process.
    """
    p = Process(target=prediction_process, args=(
        begin, end, output_dir, delay), daemon=True)
    p.start()

    return p
