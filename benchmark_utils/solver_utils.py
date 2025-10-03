from pathlib import Path
from time import perf_counter

from benchmark_utils.datasets_utils import Dataset
from fmralign import PairwiseAlignment


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

    # Time the alignment/transform process
    start_time = perf_counter()

    if dataset.target == "template_in_sample":
        group_algo.fit(dataset.dict_alignment, "template")
        dataset.dict_aligned = group_algo.transform(dataset.dict_decoding)
    elif dataset.target == "template_out_of_sample":
        dataset.dict_aligned = dict()
        for decoding_sub in dataset.dict_alignment.keys():
            # Compute a template excluding the decoding subject
            group_algo.fit(
                {
                    k: v
                    for k, v in dataset.dict_alignment.items()
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
                dataset.dict_alignment[decoding_sub], group_algo.template
            )
            dataset.dict_aligned[decoding_sub] = pairwise_algo.transform(
                dataset.dict_decoding[decoding_sub]
            )
    else:
        group_algo.fit(
            dataset.dict_alignment, dataset.dict_alignment[dataset.target]
        )
        dataset.dict_aligned = group_algo.transform(dataset.dict_decoding)

    # End the timer
    dataset.time = perf_counter() - start_time

    if solver_name.lower() == "srm":
        # Reshape arrays from (n_parcels, n_samples, n_features)
        # to (n_samples, n_parcels * n_features)
        dataset.dict_aligned = {
            k: v.transpose(1, 0, 2).reshape(v.shape[1], -1)
            for k, v in dataset.dict_aligned.items()
        }

    return dataset
