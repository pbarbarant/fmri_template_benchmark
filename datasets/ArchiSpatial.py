from benchopt import BaseDataset, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchmark_utils.datasets_utils import fetch_dataset, parse_subjects
    from pathlib import Path


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):
    # Name to select the dataset in the CLI and to display the results.
    name = "ArchiSpatial"

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    install_pip = "pip"
    requirements = []

    data_path = (
        Path("/data/parietal/store3/work/pbarbara/data/ibc/z_maps/trials")
        / name
    )

    parameters = {
        "n_parcels": [400],
        "target": ["template_in_sample", "template_out_of_sample"]
        + parse_subjects(data_path),
    }

    def get_data(self, data_path=data_path):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # The dictionary defines the keyword arguments for `Objective.set_data`

        self.dataset = fetch_dataset(
            name="IBC" + f"_{self.n_parcels}",
            subjects=parse_subjects(data_path),
            target=self.target,
            data_path=data_path,
            task=self.name,
            n_parcels=self.n_parcels,
        )

        return dict(dataset=self.dataset)
