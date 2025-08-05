from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from benchmark_utils.datasets_utils import Dataset

def compute_alignment(
    group_algo,
    pairwise_algo,
    dataset: Dataset,
    solver_name: str,
) -> Dataset:
    dataset.solver = solver_name
    output_dir = Path("outputs") / dataset.name / dataset.task_name / f"{dataset.test_sub}" / f"ext_template={str(dataset.external_template)}" / solver_name
    output_dir.mkdir(exist_ok=True, parents=True)
    dataset.output_dir = output_dir
    
    if dataset.external_template is True:
        test_sub_alignment = dataset.dict_alignment.pop(dataset.test_sub)
        # Time the alignment process
        start_time = perf_counter()
        group_algo.fit(dataset.dict_alignment, "template")
        dataset.time = perf_counter() - start_time        
        template = group_algo.template
        # Align the subjects to the template except the left-out subject
        dataset.dict_aligned = group_algo.transform({k: v for k, v in dataset.dict_decoding.items() if k != dataset.test_sub})
        
        # Align the left-out subject to the template
        pairwise_algo.fit(test_sub_alignment, template)
        left_out_aligned = pairwise_algo.transform(dataset.dict_decoding[dataset.test_sub])
        dataset.dict_aligned[dataset.test_sub] = left_out_aligned
    else:
        # Time the alignment process
        start_time = perf_counter()
        group_algo.fit(dataset.dict_alignment, "template")
        dataset.time = perf_counter() - start_time
        
        dataset.dict_aligned = group_algo.transform(dataset.dict_decoding)
    
    return dataset