from pathlib import Path
from time import perf_counter

from fmralign.alignment.utils import (
    _check_method,
    _fit_template,
    _map_to_target,
)

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

    if solver_name.lower().startswith("srm"):
        # Reshape arrays from (n_parcels, n_samples, n_features)
        # to (n_samples, n_parcels * n_features)
        fold.dict_aligned = {
            k: v.transpose(1, 0, 2).reshape(v.shape[1], -1)
            for k, v in fold.dict_aligned.items()
        }


def align_one_fold_hcp(fold: Fold, group_algo, solver_name: str) -> None:
    labels = group_algo.labels
    decoding_subjects = fold.decoding_subjects
    method = _check_method(group_algo.method)
    _, external_template = _fit_template(
        [
            v[:, labels != 0]
            for k, v in fold.dict_alignment.items()
            if k not in decoding_subjects
        ],
        method,
        labels[labels != 0],
        group_algo.n_jobs,
        group_algo.verbose,
        group_algo.n_iter,
        group_algo.scale_template,
    )
    fits = _map_to_target(
        (fold.dict_alignment[k][:, labels != 0] for k in decoding_subjects),
        external_template,
        method,
        labels[labels != 0],
        group_algo.n_jobs,
        group_algo.verbose,
    )
    fold.dict_aligned = {
        s: estimator.transform(fold.dict_decoding[s][:, labels != 0])
        for s, estimator in zip(fold.dict_decoding.keys(), fits)
    }
    if solver_name.lower().startswith("srm"):
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
        if dataset.name == "HCP":
            align_one_fold_hcp(fold, group_algo, solver_name)
        else:
            align_one_fold(fold, group_algo, dataset.target, solver_name)

    return dataset
