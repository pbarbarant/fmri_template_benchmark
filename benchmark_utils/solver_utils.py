from pathlib import Path
from time import perf_counter

from benchmark_utils.datasets_utils import Dataset, Fold


def align_one_fold(
    fold: Fold, group_algo, target: str, solver_name: str
) -> None:
    # Time the alignment/transform process
    start_time = perf_counter()

    if target == "template_in_sample":
        group_algo.fit(fold.dict_alignment, "template")
        fold.dict_aligned = group_algo.transform(fold.dict_decoding)
    elif target == "template_out_of_sample":
        group_algo.fit(fold.dict_alignment, "leave_one_subject_out")
        fold.dict_aligned = group_algo.transform(fold.dict_decoding)
    else:
        group_algo.fit(fold.dict_alignment, fold.dict_alignment[target])
        fold.dict_aligned = group_algo.transform(fold.dict_decoding)

    # End the timer
    fold.time = perf_counter() - start_time

    if solver_name.lower() == "srm":
        # Reshape arrays from (n_parcels, n_samples, n_features)
        # to (n_samples, n_parcels * n_features)
        fold.dict_aligned = {
            k: v.transpose(1, 0, 2).reshape(v.shape[1], -1)
            for k, v in fold.dict_aligned.items()
        }


def compute_alignment(
    group_algo,
    dataset: Dataset,
    solver_name: str,
) -> Dataset:
    dataset.solver_name = solver_name
    output_dir = (
        Path("outputs")
        / dataset.name
        / dataset.task_name
        / dataset.solver_name
        / dataset.target
    )
    output_dir.mkdir(exist_ok=True, parents=True)
    dataset.output_dir = output_dir

    for fold in dataset.folds:
        print(f"Aligning on fold n°{fold.index}")
        align_one_fold(fold, group_algo, dataset.target, solver_name)

    return dataset
