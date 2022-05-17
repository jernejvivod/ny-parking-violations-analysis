import dask.dataframe as dd


def read_csv(dataset_path: str) -> dd.DataFrame:
    return dd.read_csv(
        dataset_path,
        dtype={
            'Days Parking In Effect    ': str,
            'From Hours In Effect': str,
            'House Number': str,
            'Intersecting Street': str,
            'Issuer Command': str,
            'Issuer Precinct': str,
            'Issuer Squad': str,
            'Meter Number': str,
            'Time First Observed': str,
            'To Hours In Effect': str,
            'Violation Code': str,
            'Violation Description': str,
            'Violation In Front Of Or Opposite': str,
            'Violation Legal Code': str,
            'Violation Post Code': str,
        },
        assume_missing=True,
    )

def map_code_to_description(code: str):
    descriptions = {
        '36': 'School Zone Speed Violation',
        '21': 'No Parking',
        '38': 'Failure to Display Meter Rec',
        '14': 'No Standing',
        '40': 'Fire Hydrant',
        '5': 'Bus Lane Violation',
        '7': 'Red Light Violation',
        '71': 'Insp. Sticker Expired',
        '20': 'No Parking (Non-COM)',
        '70': 'Reg. Sticker Expired (NYS)',
    }
    return descriptions[code]
