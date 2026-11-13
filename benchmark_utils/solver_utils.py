from pathlib import Path
from time import perf_counter

import numpy as np
from fmralign.alignment.utils import (
    _check_method,
    _fit_template,
    _map_to_target,
)

from benchmark_utils.datasets_utils import Dataset, Fold, parse_target


def align_one_fold(
    fold: Fold, group_algo, target: str, solver_name: str, output_dir: Path
) -> None:
    # Time the alignment/transform process
    start_time = perf_counter()
    _, left_out = parse_target(target)

    if target == "template_in_sample":
        group_algo.fit(fold.dict_alignment, "template")
        fold.dict_aligned = group_algo.transform(fold.dict_decoding)
        (output_dir / f"fold_{fold.index}").mkdir(parents=True, exist_ok=True)
        np.save(
            output_dir / f"fold_{fold.index}" / "template.npy",
            group_algo.template,
        )
    elif left_out is not None:
        # Template learnt without the left-out subject, which is then mapped
        # onto this same template: all subjects share one common space.
        train = {k: v for k, v in fold.dict_alignment.items() if k != left_out}
        group_algo.fit(train, "template")
        fold.dict_aligned = group_algo.transform(
            {k: fold.dict_decoding[k] for k in train}
        )
        [estimator] = _map_to_target(
            [fold.dict_alignment[left_out]],
            group_algo.template,
            group_algo.method_,
            group_algo.labels_,
            group_algo.n_jobs,
            group_algo.verbose,
        )
        fold.dict_aligned[left_out] = estimator.transform(
            fold.dict_decoding[left_out]
        )
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


def align_one_fold_hcp(
    fold: Fold, group_algo, solver_name: str, output_dir: Path
) -> None:
    start_time = perf_counter()
    labels = group_algo.labels
    method = _check_method(group_algo.method)

    def data_generator():
        return (
            np.vstack(
                [
                    np.load(m)[np.ix_(mask, labels != 0)]
                    for m, mask in zip(movies, fold.timepoints_masks)
                ]
            )
            for movies in fold.dict_alignment.values()
        )

    fits, _ = _fit_template(
        data_generator,
        method,
        labels[labels != 0],
        group_algo.n_jobs,
        group_algo.verbose,
        group_algo.n_iter,
        group_algo.scale_template,
    )
    fold.dict_aligned = {
        s: estimator.transform(fold.dict_decoding[s][:, labels != 0])
        for s, estimator in zip(fold.dict_decoding.keys(), fits)
    }
    fold.time = perf_counter() - start_time
    np.save(
        output_dir / "template.npy",
        np.mean(list(fold.dict_aligned.values()), axis=0),
    )
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
        / dataset.target.replace(":", "/")
    )
    output_dir.mkdir(exist_ok=True, parents=True)
    dataset.output_dir = output_dir

    for fold in dataset.folds:
        print(f"Aligning on fold n°{fold.index}")
        if dataset.name == "HCP":
            align_one_fold_hcp(fold, group_algo, solver_name, output_dir)
        else:
            align_one_fold(
                fold, group_algo, dataset.target, solver_name, output_dir
            )

    return dataset
