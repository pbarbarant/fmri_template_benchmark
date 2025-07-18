from pathlib import Path

import numpy as np
from joblib import dump
from scipy.stats import pearsonr
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import (
    LeaveOneGroupOut,
    cross_val_score,
    cross_validate,
)
from sklearn.svm import LinearSVC

from benchmark_utils.conf import N_JOBS


def compute_groups(dataset):
    subject_dict = dataset.dict_aligned
    n_samples = next(iter(subject_dict.values())).shape[0]
    groups = np.concatenate(
        [np.repeat(i, n_samples) for i in range(len(subject_dict.keys()))]
    )
    return groups


def compute_X_y(dataset):
    dict_aligned = dataset.dict_aligned
    X = np.vstack(
        [dict_aligned[subject] for subject in dataset.subjects]
    )
    y = np.hstack(
        [dataset.dict_y[subject] for subject in dataset.subjects]
    )
    return X, y


def compute_pearson_corrs(dataset):
    """Compute the Pearson correlation between each subject and the target."""
    target = dataset.target_name
    parcel_masker = dataset.parcel_masker
    masker = dataset.masker
    pearson_corrs = []
    if target == "template":
        target_img = dataset.template.img
        for subject in dataset.subjects:
            # Do not compare a subject with itself
            if subject != target:
                subject_img = dataset.dict_aligned[subject].img
                subject_corr = pearson_corr_parcels(
                    subject_img, target_img, parcel_masker
                )
                pearson_corrs.append(subject_corr)
    else:
        target_img = dataset.dict_aligned[target].img
        target_data = masker.transform(target_img)
        avg_data_all_subjects = np.zeros_like(target_data)
        for subject in dataset.subjects:
            subject_data = masker.transform(dataset.dict_aligned[subject].img)
            avg_data_all_subjects += subject_data / len(dataset.subjects)

        avg_img_all_subjects = masker.inverse_transform(avg_data_all_subjects)
        pearson_corrs = [
            pearson_corr_parcels(
                avg_img_all_subjects, target_img, parcel_masker
            ),
        ]
    return pearson_corrs


def pearson_corr(data1, data2):
    # Z-score the time series along time axis (axis=0)
    data1_z = (data1 - data1.mean(axis=0)) / data1.std(axis=0)
    data2_z = (data2 - data2.mean(axis=0)) / data2.std(axis=0)

    # Compute element-wise product and average across time points
    corr = np.nanmean(data1_z * data2_z, axis=0)

    # Return average correlation across parcels
    return np.nanmean(corr)


def pearson_corr_parcels(img1, img2, parcel_masker):
    """Compute the Pearson correlation between two images
    by averaging the signal in each parcel."""
    n_samples = img1.shape[-1]
    parceled_data1, parceled_data2 = parcel_masker.transform([img1, img2])
    data1, data2 = parceled_data1.to_list(), parceled_data2.to_list()
    correlations = np.zeros((len(data1), n_samples))
    for i, (d1, d2) in enumerate(zip(data1, data2)):
        for j in range(n_samples):
            correlations[i, j] = pearsonr(d1[j, :], d2[j, :])[0]

    # Remove NaN values
    cleaned_correlations = np.nan_to_num(correlations)
    return np.mean(cleaned_correlations)


def save_weights(estimator, dataset, subject=None):
    output_dir = (
        Path("outputs") / dataset.name / dataset.solver / dataset.target_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save the weights of the estimator
    np.save(output_dir / f"{subject}_weights.npy", estimator.coef_)
    weights_labels = estimator.classes_
    # Save the labels of the weights as csv
    np.savetxt(
        output_dir / f"{subject}_weights_labels.csv", weights_labels, fmt="%s"
    )


def evaluate_task_dataset(dataset, max_iter=1000):
    # Leave one subject out cross-validation
    groups = compute_groups(dataset)
    X, y = compute_X_y(dataset)

    svc = LinearSVC(max_iter=max_iter)
    scores = cross_validate(
        svc,
        X,
        y,
        groups=groups,
        cv=LeaveOneGroupOut(),
        n_jobs=N_JOBS,
        return_estimator=True,
    )
    cv_scores_classif = scores["test_score"]
    for i, estimator in enumerate(scores["estimator"]):
        save_weights(estimator, dataset, subject=dataset.subjects[i])

    avg_score = np.mean(cv_scores_classif)
    chance_level = np.mean(
        cross_val_score(
            DummyClassifier(strategy="most_frequent"),
            X,
            y,
            groups=groups,
            cv=LeaveOneGroupOut(),
            n_jobs=N_JOBS,
        )
    )

    print(f"Average decoding accuracy: {avg_score:.2f}")

    return avg_score, chance_level, cv_scores_classif




def evaluate_dataset(dataset, max_iter=1000):
    # Compute the Pearson correlations for the dataset
    # pearson_corrs = compute_pearson_corrs(dataset)
    # Evaluate the decoding performance
    pearson_corrs = []
    avg_score, chance_level, cv_scores_classif = evaluate_task_dataset(
        dataset, max_iter=max_iter
    )
    # Save the results
    save_decoding_results(
        dataset,
        avg_score,
        chance_level,
        cv_scores_classif,
        pearson_corrs,
    )

    # Return only the average score for benchopt
    return avg_score


def save_decoding_results(
    dataset,
    avg_score,
    chance_level,
    cv_scores_classif,
    pearson_corrs,
):
    output_dir = (
        Path("outputs") / dataset.name / dataset.solver / dataset.target_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dict = {
        "avg_score": avg_score,
        "chance_level": chance_level,
        "cv_scores_classif": cv_scores_classif,
        "pearson_corrs": pearson_corrs,
        "time": dataset.time,
    }
    # Dump the results with joblib
    dump(results_dict, output_dir / "decoding_results.pkl")
    print(f"Decoding results saved in {output_dir}")
