from pathlib import Path
from time import perf_counter

from benchmark_utils.datasets_utils import Dataset, Fold
from fmralign import PairwiseAlignment


def align_one_fold(
    fold: Fold, group_algo, target: str, solver_name: str
) -> None:
    # Time the alignment/transform process
    start_time = perf_counter()

    if target == "template_in_sample":
        group_algo.fit(fold.dict_alignment, "template")
        fold.dict_aligned = group_algo.transform(fold.dict_decoding)
    elif target == "template_out_of_sample":
        fold.dict_aligned = dict()
        for decoding_sub in fold.dict_alignment.keys():
            # Compute a template excluding the decoding subject
            group_algo.fit(
                {
                    k: v
                    for k, v in fold.dict_alignment.items()
                    if k != "decoding_sub"
                },
                "template",
            )
            # Align the decoding subject using a pairwise estimator
            pairwise_algo = PairwiseAlignment(
                method=group_algo.method,
                labels=group_algo.labels,
                n_jobs=group_algo.n_jobs,
            )
            pairwise_algo.fit(
                fold.dict_alignment[decoding_sub], group_algo.template
            )
            fold.dict_aligned[decoding_sub] = pairwise_algo.transform(
                fold.dict_decoding[decoding_sub]
            )
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
