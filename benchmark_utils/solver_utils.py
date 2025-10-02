from pathlib import Path
from time import perf_counter

from benchmark_utils.datasets_utils import Dataset


def compute_alignment(
    group_algo,
    dataset: Dataset,
    solver_name: str,
) -> Dataset:
    dataset.solver_name = solver_name
    output_dir = (
        Path("outputs") / dataset.name / dataset.task_name / solver_name
    )
    output_dir.mkdir(exist_ok=True, parents=True)
    dataset.output_dir = output_dir
    # Time the alignment process
    start_time = perf_counter()
    group_algo.fit(dataset.dict_alignment, "template")
    dataset.time = perf_counter() - start_time

    dataset.dict_aligned = group_algo.transform(dataset.dict_decoding)

    if solver_name.lower() == "srm":
        # Reshape arrays from (n_parcels, n_samples, n_features)
        # to (n_samples, n_parcels * n_features)
        dataset.dict_aligned = {
            k: v.transpose(1, 0, 2).reshape(v.shape[1], -1)
            for k, v in dataset.dict_aligned.items()
        }

    return dataset
