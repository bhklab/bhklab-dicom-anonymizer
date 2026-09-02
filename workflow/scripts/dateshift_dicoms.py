import datetime
import logging
from pathlib import Path

import pandas as pd
import pydicom

logger = logging.getLogger()

def apply_date_shift(
        ds:pydicom.FileDataset,
        date_tags:list,
        pat_offset:int,
    ) -> pydicom.FileDataset:
    """Applies the date shift to all date tags in list of date_tags for a single DICOM file"""

    for tag_name in date_tags:
        # Try to access the DICOM tag; if it doesn't exist, log the error and continue
        # Not every DICOM will have all of the date tags, so we need to handle missing tags gracefully
        if tag_name in ds:
            original_date = ds.data_element(tag_name).value

            if 'Time' in tag_name:
                format = "%Y%m%d%H%M%S"
                original_date = original_date[:-5]
            else:
                format = "%Y%m%d"
            
            date_obj = datetime.datetime.strptime(original_date, format)
            new_date = (date_obj - datetime.timedelta(days=pat_offset)).strftime(format)
            ds.data_element(tag_name).value = new_date

    return ds


def calculate_date_offset(
        original_date_str: str,
        anchor_date_str: str = "19010101"
    ) -> int:
    """Calculate days between original date and anchor date. Returns positive integer."""
    return (datetime.datetime.strptime(original_date_str, '%Y%m%d') - datetime.datetime.strptime(anchor_date_str, '%Y%m%d')).days





