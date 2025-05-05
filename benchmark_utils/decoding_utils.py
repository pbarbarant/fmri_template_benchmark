from pathlib import Path

import numpy as np
from joblib import Parallel, delayed, dump
from nilearn import image
from nilearn.maskers._utils import concatenate_surface_images
from scipy.stats import pearsonr
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import (
    LeaveOneGroupOut,
    cross_val_score,
    cross_validate,
)

from benchmark_utils.conf import N_JOBS


def compute_groups(subject_dict):
    n_samples = next(iter(subject_dict.values())).img.shape[-1]
    groups = np.concatenate(
        [np.repeat(i, n_samples) for i in range(len(subject_dict.keys()))]
    )
    return groups


def compute_batched_groups(subject_dict, n_groups=10):
    n_samples = next(iter(subject_dict.values())).img.shape[-1]
    n_subjects = len(subject_dict.keys())
    batch_len = n_samples * n_subjects // n_groups
    groups = np.concatenate([np.repeat(i, batch_len) for i in range(n_groups)])
    return groups


def compute_X_y(dataset):
    dict_aligned = dataset.dict_aligned
    masker = dataset.masker
    if dataset.is_surf:
        imgs = concatenate_surface_images(
            [dict_aligned[subject].img for subject in dataset.subjects]
        )
    else:
        imgs = image.concat_imgs(
            [dict_aligned[subject].img for subject in dataset.subjects]
        )
    X = masker.transform(imgs)
    y = np.concatenate(
        [dict_aligned[subject].y for subject in dataset.subjects]
    )
    return X, y


def compute_pearson_corrs(dataset):
    """Compute the Pearson correlation between each subject and the target."""
    target = dataset.target
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
            subject_data = masker.transform(
                dataset.dict_aligned[subject].img
            )
            avg_data_all_subjects += subject_data / len(dataset.subjects)
        
        avg_img_all_subjects = masker.inverse_transform(avg_data_all_subjects)
        pearson_corrs = [
            pearson_corr_parcels(
                avg_img_all_subjects, target_img, parcel_masker
            ),
        ]
    return pearson_corrs


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
    masker = dataset.masker
    output_dir = (
        Path("outputs") / dataset.name / dataset.solver / dataset.target
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_img = masker.inverse_transform(estimator.coef_)
    weights_labels = estimator.classes_
    weights_img.to_filename(output_dir / f"{subject}_weights.nii.gz")
    # Save the labels of the weights as csv
    np.savetxt(
        output_dir / f"{subject}_weights_labels.csv", weights_labels, fmt="%s"
    )


def evaluate_task_dataset(dataset, max_iter=1000):
    # Leave one subject out cross-validation
    if dataset.name.lower().startswith("hcp"):
        groups = compute_batched_groups(dataset.dict_aligned, n_groups=10)
    else:
        groups = compute_groups(dataset.dict_aligned)

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


def classify_subject_movie(template_img, img, y, parcel_masker):
    segments_ids = np.unique(y)
    res = []
    # Get the list of segments for the template
    segments_template = []
    segments_subject = []
    for segment_id in segments_ids:
        # Get the slice of indices corresponding to the segment
        segment_slice = np.where(y == segment_id)[0]
        # Get the image segment for the template
        segments_template.append(image.index_img(template_img, segment_slice))
        # Get the image segment for the subject
        segments_subject.append(image.index_img(img, segment_slice))

    for i in range(len(segments_ids)):
        # For each movie segment of the subject
        segment_subject = segments_subject[i]
        correct_id = segments_ids[i]
        corr_list = []
        for j in range(len(segments_ids)):
            segment_template = segments_template[j]
            # Compute the Pearson correlation with each segment of the template
            corr_list.append(
                pearson_corr_parcels(
                    segment_template, segment_subject, parcel_masker
                )
            )
        # Predict the segment with the highest correlation
        predicted_id = segments_ids[np.argmax(corr_list)]
        res.append(predicted_id == correct_id)
    return np.mean(res)


def evaluate_movie_dataset(dataset):
    dict_aligned = dataset.dict_aligned
    masker = dataset.masker

    parcel_masker = dataset.parcel_masker

    # Parallelize the classification of each subject
    cv_scores_classif = Parallel(n_jobs=N_JOBS, verbose=11)(
        delayed(classify_subject_movie)(
            dataset.template.img,
            dict_aligned[subject].img,
            dataset.template.y,
            parcel_masker,
        )
        for subject in dataset.subjects
    )

    avg_score = np.mean(cv_scores_classif)
    chance_level = 1 / len(dict_aligned[dataset.subjects[0]].y)

    print(f"Average decoding accuracy: {avg_score:.2f}")
    print(f"Chance level: {chance_level:.2f}")

    return avg_score, chance_level, cv_scores_classif


def evaluate_template_dataset(dataset, max_iter=1000):
    # Compute the voxel-wise pearson correlation between all subjects
    # and the template
    pearson_corrs = compute_pearson_corrs(dataset)

    # Evaluate the decoding performance
    if dataset.paradigm == "movie":
        avg_score, chance_level, cv_scores_classif = evaluate_movie_dataset(
            dataset
        )
    else:
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


def evaluate_subject_dataset(dataset, max_iter=1000):
    masker = dataset.masker
    dict_aligned = dataset.dict_aligned
    # Compute the voxel-wise pearson correlation between all subjects
    # and the target
    pearson_corrs = compute_pearson_corrs(dataset)

    # Evaluate the decoding performance
    svc = LinearSVC(max_iter=max_iter)
    X_train = masker.transform(
        image.concat_imgs(
            [
                dict_aligned[subject].img
                for subject in dataset.subjects
                if subject != dataset.target
            ]
        )
    )

    y_train = np.concatenate(
        [
            dict_aligned[subject].y
            for subject in dataset.subjects
            if subject != dataset.target
        ]
    )

    X_test = masker.transform(dict_aligned[dataset.target].img)
    y_test = dict_aligned[dataset.target].y

    svc.fit(X_train, y_train)
    save_weights(svc, dataset, subject=dataset.target)
    avg_score = svc.score(X_test, y_test)

    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_train)
    chance_level = dummy_clf.score(X_test, y_test)

    cv_scores_classif = [avg_score]
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
        Path("outputs") / dataset.name / dataset.solver / dataset.target
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
