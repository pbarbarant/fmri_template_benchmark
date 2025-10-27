import numpy as np
from joblib import dump
from sklearn.model_selection import cross_validate, LeaveOneGroupOut
from sklearn.svm import LinearSVC
from typing import List
from benchmark_utils.conf import N_JOBS
from benchmark_utils.datasets_utils import Dataset, Fold
from nilearn.maskers import NiftiMasker
from pathlib import Path


def save_weights(scores: dict, masker: NiftiMasker, output_dir: Path):
    estimators = scores["estimator"]
    classes_ = estimators[0].classes_
    coefs_aggregated = np.mean(
        np.stack([estimator.coef_ for estimator in estimators]),
        axis=0,
    )

    for i, class_ in enumerate(classes_):
        masker.inverse_transform(coefs_aggregated[i]).to_filename(
            output_dir / f"coefs_{class_}.nii.gz"
        )


def decode_one_fold(
    fold: Fold,
    subjects: List[str],
    target: str,
    solver_name: str,
    masker: NiftiMasker,
    output_dir: Path,
    max_iter: int = 1000,
):
    svc = LinearSVC(max_iter=max_iter)
    # Cross decoding in the case of the template
    if target == "template_in_sample" or target == "template_out_of_sample":
        X = np.vstack([fold.dict_aligned[sub] for sub in subjects])
        y = np.hstack([fold.dict_y[sub] for sub in subjects])
        groups = np.concatenate(
            [[i] * len(fold.dict_y[sub]) for i, sub in enumerate(subjects)]
        )
        scores = cross_validate(
            svc,
            X,
            y,
            cv=LeaveOneGroupOut(),
            groups=groups,
            return_estimator=True,
            n_jobs=N_JOBS,
            verbose=1,
        )
        cv_scores = scores["test_score"].tolist()
        chance_level = 1 / len(np.unique(y))
        if solver_name.lower() != "srm":
            save_weights(scores, masker, output_dir)
        print(f"Average decoding accuracy: {np.mean(cv_scores):.2f}")
    # Decode the target in the pairwise case
    else:
        X_train = np.vstack(
            [fold.dict_aligned[sub] for sub in subjects if sub != target]
        )
        y_train = np.hstack(
            [fold.dict_y[sub] for sub in subjects if sub != target]
        )
        X_test = fold.dict_aligned[target]
        y_test = fold.dict_y[target]
        svc.fit(X_train, y_train)
        cv_scores = [svc.score(X_test, y_test)]
        chance_level = 1 / len(np.unique(y_test))
        print(f"Decoding accuracy on {target} : {cv_scores[0]:.2f}")
    return cv_scores, chance_level


def evaluate_dataset(dataset: Dataset, max_iter=1000):
    # Compute the Pearson correlations for the dataset
    # pearson_corrs = compute_pearson_corrs(dataset)
    # Evaluate the decoding performance
    all_scores = []
    for fold in dataset.folds:
        fold_output_dir = dataset.output_dir / f"fold_{fold.index}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        cv_scores, chance_level = decode_one_fold(
            fold,
            subjects=dataset.subjects,
            target=dataset.target,
            solver_name=dataset.solver_name,
            masker=dataset.masker,
            output_dir=fold_output_dir,
            max_iter=max_iter,
        )
        all_scores.extend(cv_scores)
        # Save the results
        results_dict = {
            "cv_scores": cv_scores,
            "dataset_name": dataset.name,
            "task_name": dataset.task_name,
            "chance_level": chance_level,
            "time": fold.time,
            "target": dataset.target,
            "fold": fold.index,
            "solver_name": dataset.solver_name,
            "subject": dataset.subjects
            if len(cv_scores) > 1
            else dataset.target,
        }
        # Dump the results with joblib
        dump(results_dict, fold_output_dir / "decoding_results.pkl")
        print(f"Decoding results saved in {fold_output_dir}")

    # Return only the average score for benchopt
    return np.mean(all_scores)
