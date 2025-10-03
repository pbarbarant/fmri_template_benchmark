import numpy as np
from joblib import dump
from sklearn.model_selection import cross_validate, LeaveOneGroupOut
from sklearn.svm import LinearSVC

from benchmark_utils.conf import N_JOBS
from benchmark_utils.datasets_utils import Dataset


def save_weights(scores: dict, dataset: Dataset):
    for subject in dataset.subjects:
        # Locate the subject index in all estimators.classes_
        map_indices = [
            estimator.classes_.tolist().index(subject)
            for estimator in scores["estimator"]
        ]
        # Extract the corresponding weights for the subject
        weights = np.mean(
            [
                estimator.coef_[index]
                for estimator, index in zip(scores["estimator"], map_indices)
            ],
            axis=0,
        )
        dataset.masker.inverse_transform(weights).to_filename(
            dataset.output_dir / f"{subject}_weights.nii.gz"
        )


def decode(dataset: Dataset, max_iter: int = 1000):
    svc = LinearSVC(max_iter=max_iter)
    # Cross decoding in the case of the template
    if (
        dataset.target == "template_in_sample"
        or dataset.target == "template_out_of_sample"
    ):
        X = np.vstack([dataset.dict_aligned[sub] for sub in dataset.subjects])
        y = np.hstack([dataset.dict_y[sub] for sub in dataset.subjects])
        groups = np.concatenate(
            [
                [i] * len(dataset.dict_y[sub])
                for i, sub in enumerate(dataset.subjects)
            ]
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
        # save_weights(scores, dataset)
        print(f"Average decoding accuracy: {np.mean(cv_scores):.2f}")
    # Decode the target in the pairwise case
    else:
        X_train = np.vstack(
            [
                dataset.dict_aligned[sub]
                for sub in dataset.subjects
                if sub != dataset.target
            ]
        )
        y_train = np.hstack(
            [
                dataset.dict_y[sub]
                for sub in dataset.subjects
                if sub != dataset.target
            ]
        )
        X_test = dataset.dict_aligned[dataset.target]
        y_test = dataset.dict_y[dataset.target]
        svc.fit(X_train, y_train)
        cv_scores = [svc.score(X_test, y_test)]
        chance_level = 1 / len(np.unique(y_test))
        print(f"Decoding accuracy on {dataset.target} : {cv_scores[0]:.2f}")
    return cv_scores, chance_level


def evaluate_dataset(dataset: Dataset, max_iter=1000):
    # Compute the Pearson correlations for the dataset
    # pearson_corrs = compute_pearson_corrs(dataset)
    # Evaluate the decoding performance
    cv_scores, chance_level = decode(dataset, max_iter=max_iter)
    # Save the results
    save_decoding_results(
        dataset,
        cv_scores,
        chance_level,
    )

    # Return only the average score for benchopt
    return np.mean(cv_scores)


def save_decoding_results(
    dataset: Dataset,
    cv_scores: np.ndarray,
    chance_level: float,
):
    results_dict = {
        "cv_scores": cv_scores,
        "dataset_name": dataset.name,
        "task_name": dataset.task_name,
        "chance_level": chance_level,
        "time": dataset.time,
        "target": dataset.target,
    }
    # Dump the results with joblib
    dump(results_dict, dataset.output_dir / "decoding_results.pkl")
    print(f"Decoding results saved in {dataset.output_dir}")
