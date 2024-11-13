from pathlib import Path
import pyarrow.parquet as pq
from dataclasses import dataclass
import pandas as pd


@dataclass
class Case:
    info: pd.DataFrame
    measurements: pd.DataFrame


class RawDataset:
    def __init__(
        self,
        dataset_root: Path | None = None,
        unit: str = "VG4",
        load_training: bool = False,
        load_synthetic: bool = False,
    ) -> None:

        if dataset_root is None:
            dataset_root = Path.cwd() / "Dataset"

        cases = {
            "test": [
                dataset_root / f"{unit}_generator_data_testing_real_measurements.parquet",
                dataset_root / f"{unit}_generator_data_testing_real_info.csv",
            ],
        }

        if load_training:
            cases = {
                **cases,
                "train": [
                    dataset_root / f"{unit}_generator_data_training_measurements.parquet",
                    dataset_root / f"{unit}_generator_data_training_info.csv",
                ],
            }

        if load_synthetic:
            cases = {
                **cases,
                "test_s01": [
                    dataset_root / f"{unit}_generator_data_testing_synthetic_01_measurements.parquet",
                    dataset_root / f"{unit}_generator_data_testing_synthetic_01_info.csv",
                ],
                "test_s02": [
                    dataset_root / f"{unit}_generator_data_testing_synthetic_02_measurements.parquet",
                    dataset_root / f"{unit}_generator_data_testing_synthetic_02_info.csv",
                ],
            }

        self.data_dict = dict()

        for case_label, (measurements_file, info_file) in cases.items():
            info = pd.read_csv(info_file)
            measurements = pq.read_table(dataset_root / measurements_file).to_pandas()
            self.data_dict[case_label] = Case(info, measurements)

    @staticmethod
    def read_parquet_schema_df(uri: str) -> pd.DataFrame:
        """Return a Pandas dataframe corresponding to the schema of a local URI of a parquet file.

        The returned dataframe has the columns: column, pa_dtype
        """
        # Ref: https://stackoverflow.com/a/64288036/
        schema = pq.read_schema(uri, memory_map=True)
        schema = pd.DataFrame(
            ({"column": name, "pa_dtype": str(pa_dtype)} for name, pa_dtype in zip(schema.names, schema.types))
        )
        schema = schema.reindex(
            columns=["column", "pa_dtype"], fill_value=pd.NA
        )  # Ensures columns in case the parquet file has an empty dataframe.
        return schema
