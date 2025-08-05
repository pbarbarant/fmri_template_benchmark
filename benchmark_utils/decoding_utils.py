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
    # Save the weights of the estimator
    np.save(dataset.output_dir / f"{subject}_weights.npy", estimator.coef_)
    weights_labels = estimator.classes_
    # Save the labels of the weights as csv
    np.savetxt(
        dataset.output_dir / f"{subject}_weights_labels.csv", weights_labels, fmt="%s"
    )


def evaluate_task_dataset(dataset, max_iter=1000):
    svc = LinearSVC(max_iter=max_iter)
    dummy = DummyClassifier(strategy="most_frequent")
    X_train = np.vstack(
        [dataset.dict_aligned[sub] for sub in dataset.subjects if sub != dataset.test_sub]
    )
    y_train = np.hstack(
        [dataset.dict_y[sub] for sub in dataset.subjects if sub != dataset.test_sub]
    )
    X_test = dataset.dict_aligned[dataset.test_sub]
    y_test = dataset.dict_y[dataset.test_sub]

    svc.fit(X_train, y_train)
    dummy.fit(X_train, y_train)
    score = svc.score(X_test, y_test)
    chance_level = dummy.score(X_test, y_test)
    save_weights(svc, dataset, subject=dataset.test_sub)

    print(f"Decoding accuracy on {dataset.test_sub}: {score:.2f}")

    return score, chance_level


def evaluate_dataset(dataset, max_iter=1000):
    # Compute the Pearson correlations for the dataset
    # pearson_corrs = compute_pearson_corrs(dataset)
    # Evaluate the decoding performance
    score, chance_level = evaluate_task_dataset(
        dataset, max_iter=max_iter
    )
    # Save the results
    save_decoding_results(
        dataset,
        score,
        chance_level,
    )

    # Return only the classification score for benchopt
    return score


def save_decoding_results(
    dataset,
    score,
    chance_level,
):
    results_dict = {
        "score": score,
        "dataset_name": dataset.name,
        "task_name": dataset.task_name,
        "chance_level": chance_level,
        "time": dataset.time,
        "test_sub": dataset.test_sub,
        "external_template": str(dataset.external_template),
    }
    # Dump the results with joblib
    dump(results_dict, dataset.output_dir / "decoding_results.pkl")
    print(f"Decoding results saved in {dataset.output_dir}")
