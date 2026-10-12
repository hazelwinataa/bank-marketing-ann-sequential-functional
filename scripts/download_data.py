"""Download the UCI Bank Marketing file required by the original notebook.

This utility is a new repository helper; it is not part of the original ANN experiment.
The source data license is CC BY 4.0. See data/README.md for attribution.
"""

# STANDARD LIBRARY
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile
import csv

DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
TARGET = Path(__file__).resolve().parents[1] / "notebooks" / "bank-full.csv"


def extract_dataset(data: bytes) -> bytes:
    """Find bank-full.csv in an outer UCI ZIP or nested bank.zip."""
    with ZipFile(BytesIO(data)) as outer:
        for name in outer.namelist():
            if Path(name).name.lower() == "bank-full.csv":
                return outer.read(name)
        for name in outer.namelist():
            if Path(name).name.lower() == "bank.zip":
                with ZipFile(BytesIO(outer.read(name))) as nested:
                    for inner_name in nested.namelist():
                        if Path(inner_name).name.lower() == "bank-full.csv":
                            return nested.read(inner_name)
    raise ValueError("The official UCI archive did not contain bank-full.csv")


def main() -> None:
    print("Downloading dataset from UCI Machine Learning Repository...")
    request = Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0 academic-reproducibility"})
    with urlopen(request, timeout=45) as response:
        data = extract_dataset(response.read())
    decoded = data.decode("utf-8-sig")
    reader = csv.reader(decoded.splitlines(), delimiter=";")
    header = next(reader)
    if len(header) != 17 or "duration" not in header or "y" not in header:
        raise ValueError("Unexpected bank-full.csv header; download was not saved")
    if sum(1 for _ in reader) != 45211:
        raise ValueError("Unexpected row count: expected 45,211 bank-full.csv observations")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_bytes(data)
    print(f"Saved 45,211 observations to: {TARGET}")
    print("Dataset attribution: Moro, Rita & Cortez (2014), UCI Bank Marketing, CC BY 4.0")


if __name__ == "__main__":
    main()
