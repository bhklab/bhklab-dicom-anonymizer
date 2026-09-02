import logging
import numpy as np
import pandas as pd

from damply import dirs
from joblib import Parallel, delayed
from pathlib import Path
from pydicom.filereader import dcmread
from tqdm import tqdm

from dateshift_dicoms import calculate_date_offset, apply_date_shift
from utils.loaders import load_anon_key_yaml, load_id_map, load_crawl_db
from utils.anonymization_keys import AnonymizationKeys


HASH_PREFIX = np.abs(hash("PMCC"))

logging.basicConfig(
	level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=f'{dirs.LOGS}/apply_anonymize.log'
)

logger = logging.getLogger(__name__)


def anonymize(
        dicom_file_path:Path, 
        anon_keys:AnonymizationKeys,
        anon_id_map:dict[str:str],
        save_dir_path:Path,
        date_offset_days:int = 100,
        hash_truncate_length:int = 64,
    ) -> None:
    
    ds = dcmread(dicom_file_path)

    # remove private tags first
    ds.remove_private_tags()

    # Map PatientID and PatientName to value in id_map
    try:
        anon_id = anon_id_map[ds.data_element("PatientID").value]
    except KeyError as ke:
        msg = f"PatientID in {dicom_file_path} does not exist in the anon_id_map: {anon_id_map}. PatientID: {ds.data_element("PatientID").value}"
        logger.error(msg)
        raise ke
    
    for keyword in ["PatientID", "PatientName"]:
        ds.data_element(keyword).value = anon_id

    # remove tags in the remove list
    logger.info("Removing specified DICOM tags.")
    for keyword in anon_keys.remove:
        if keyword in ds:
            
            delattr(ds, keyword)

    logger.info("Emptying specified DICOM tags (setting to blank).")
    for keyword in anon_keys.empty:
        if keyword in ds:
            ds.data_element(keyword).value = ""

    logger.info("Hashing and truncating specified DICOM tags")
    for keyword in anon_keys.hash:
        if keyword in ds:
            hash_keyword = np.abs(hash(ds.data_element(keyword).value))
            complete_hash_value = f"{HASH_PREFIX}.{hash_keyword}"
            trunc_hash_value = complete_hash_value[:hash_truncate_length]
            ds.data_element(keyword).value = trunc_hash_value

    logger.info("Shifting dates of specified DICOM tags")
    ds = apply_date_shift(ds, anon_keys.increment_date, date_offset_days)

    anon_series_uid = ds.data_element('SeriesInstanceUID').value[-8:]

    modality = ds.data_element('Modality').value
    if modality in ['CT', 'MR', 'PT']:
        instance_number = ds.data_element('InstanceNumber').value
    else:
        instance_number = 1
    
    # if modality in ['RTSTRUCT', 'SEG']:
    #     hash_ref_uid = np.abs(hash(ds.data_element('ReferencedSeriesUID').value))
    #     complete_hash_ref_uid = f"{HASH_PREFIX}.{hash_ref_uid}"
    #     trunc_hash_ref_uid = complete_hash_ref_uid[:hash_truncate_length]
    #     ds.data_element('ReferencedSeriesUID').value = trunc_hash_ref_uid


    save_file_path = save_dir_path / anon_id / f"{modality}_{anon_series_uid}" / f"{instance_number:04d}.dcm"
    save_file_path.parent.mkdir(parents=True, exist_ok=True)

    ds.save_as(save_file_path)
   
    return ds


def proc_one(
    dicom_group:pd.DataFrame,
    image_dir:Path,
    anon_keys:AnonymizationKeys,
    anon_id_map:dict,
    save_dir_path:Path,
    scan_modality:str = 'CT',
    anchor_date:str = "19010101"
    ) -> int:
    pat_id = dicom_group['PatientID'].iloc[0]
    logger.info(f"Processing patient {pat_id}...")

    # Get the AcquisitionDate from the first scan of the specified modality for this patient to calculate the offset.  
    scan_info = dicom_group[dicom_group['Modality'] == scan_modality]
    if scan_info.empty:
        # If no scan of the specified modality is found, skip this patient and log a warning.
        logger.warning(f"No {scan_modality} scan found for patient {pat_id}. Skipping.")
        return
    
    # Get earliest acquisition date for this patient to calculate all other date offsets from
    first_scan_date = scan_info['AcquisitionDate'].min()
    # Calculate the offset using the first CT scan's AcquisitionDate, returns int 
    pat_offset_days = calculate_date_offset(first_scan_date, anchor_date)

    pat_image_dir = Path(dicom_group['folder'].iloc[0]).parent
    pat_dir_path = image_dir / pat_image_dir

    # Get anonymized patient ID for this group 
    # anonymize function is expecting a dictionary with {real_id: anon_id}
    pat_anon_id = {pat_id: anon_id_map[pat_id]}

    for dcm_file in pat_dir_path.rglob("*.dcm"):
        ds = anonymize(dcm_file, 
                  anon_keys, 
                  pat_anon_id,
                  date_offset_days=pat_offset_days,
                  save_dir_path=save_dir_path
                  )

    return 


def main(
    image_dir:Path,
    crawl_db_path:Path,
    anon_keys_path:Path,
    id_map_path:Path,
    save_dir_path:Path,
    scan_modality:str = 'CT',
    anchor_date:str = "19010101",
    parallel:bool = True,
    jobs:int = -1
    ) -> None:
    """Main function to anonymize DICOM files
       The date shift is calculated for each patient based on the AcquisitionDate of the first scan of the specified modality (default: CT) and shifting that date to an anchor date of 1901-01-01. The same offset in days is then applied to all date tags in the DICOM files for that patient.
    
    Arguments
    ---------
    image_dir: Path
        Path to the directory containing the DICOM images.
    crawl_db_path: Path
        Path to the med-imagetools index output crawl_db.json file.
    anon_keys_file: Path
        Path to the yaml file with anonymous key configuration.
    id_map_path: Path
        Path to a csv containing anonymized PatientIDs mapped to original PatientIDs
    save_dir_path: Path
        Path to save out the anonymized DICOMs
    scan_modality: str, optional
        The scan modality to use for calculating the date offset (default: CT). This should be a modality that is present for all patients in the dataset and is expected to have an AcquisitionDate tag that can be used for the date shift calculation.
    anchor_date: str, optional
        Optional string specifying the base date to adjust all images around. 
    parallel: bool, optional
        Optional flag to run date shifting in parallel. Default is True.
    jobs: int, optional
        Optional number of jobs to give parallel processor.
   
    Returns
    -------
    None
    """
    anon_keys = load_anon_key_yaml(anon_keys_path)
    id_map = load_id_map(id_map_path)
    crawl_df = load_crawl_db(crawl_db_path)

    if parallel:
        Parallel(n_jobs=jobs)(
            delayed(proc_one)(
                dicom_group=dicom_group,
                image_dir=image_dir,
                anon_keys=anon_keys,
                anon_id_map=id_map,
                save_dir_path=save_dir_path,
                scan_modality=scan_modality,
                anchor_date=anchor_date
            )
            for _pat_id, dicom_group in tqdm(
                crawl_df.groupby('PatientID'),
                desc="Applying anonymization to patient DICOMS",
                total=crawl_df['PatientID'].nunique()
            )
        )
    else:
        [
            proc_one(
                dicom_group=dicom_group,
                image_dir=image_dir,
                anon_keys=anon_keys,
                anon_id_map=id_map,
                save_dir_path=save_dir_path,
                scan_modality=scan_modality,
                anchor_date=anchor_date
            )
            for _pat_id, dicom_group in tqdm(
                crawl_df.groupby('PatientID'),
                desc="Applying anonymization to patient DICOMS",
                total=crawl_df['PatientID'].nunique()
            )
        ]

    return


if __name__ == '__main__':

    scan_modality = "CT"
    image_dir = dirs.PROCDATA / "UHN_LYM-PET"
    save_dir_path = dirs.PROCDATA / "UHN_LYM-PET" / "anon_dicoms"

    anon_keys_path = dirs.CONFIG / "anon_tcia_keys_no_hash.yaml"

    id_map_path = dirs.PROCDATA / "UHN_LYM-PET" / "RMP_LYM-PET-anon-key.csv"

    crawl_db_path = dirs.PROCDATA / ".imgtools" / "sorted_RMP_LYM_PET_DICOMS" / "crawl_db.json"


    main(
        image_dir = image_dir,
        crawl_db_path = crawl_db_path,
        anon_keys_path = anon_keys_path,
        id_map_path = id_map_path,
        save_dir_path = save_dir_path,
        scan_modality = scan_modality,
        anchor_date = "19010101",
        parallel = True,
        jobs = -1
    )
    