import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import (
    LeaveOneGroupOut,
    cross_val_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


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


def compute_X_y(subject_dict, masker):
    subject_list = list(subject_dict.keys())
    X = np.concatenate(
        [masker.transform(subject_dict[subject].img) for subject in subject_list]
    )
    y = np.concatenate([subject_dict[subject].y for subject in subject_list])
    return X, y


def compute_pearson_corr(dataset):
    template_img = dataset.template.img
    masker = dataset.masker
    avg_parcel_template = masker.transform(template_img)
    pearson_corrs = []
    for subject in dataset.subjects:
        subject_img = dataset.dict_aligned[subject].img
        subject_img = masker.transform(subject_img)
        subject_corr = np.corrcoef(subject_img, template_img)
        pearson_corrs.append(subject_corr)
    return pearson_corrs


def evaluate_dataset(dataset, max_iter=100):
    dict_aligned = dataset.dict_aligned
    masker = dataset.masker

    # Compute the voxel-wise pearson correlation between all subjects
    # and the template
    # pearson_corrs = compute_pearson_corr(dataset)
    pearson_corrs = [1.0] * len(dataset.subjects)

    # Create cross-validation object on each subject
    pipeline_svc = make_pipeline(
        StandardScaler(),
        LinearSVC(max_iter=int(max_iter), penalty="l2"),
    )
    X, y = compute_X_y(dict_aligned, masker)
    if dataset.name.lower().startswith("hcp"):
        groups = compute_batched_groups(dict_aligned, n_groups=10)
    else:
        groups = compute_groups(dict_aligned)

    cv_scores_svc = cross_val_score(
        pipeline_svc,
        X,
        y,
        groups=groups,
        cv=LeaveOneGroupOut(),
    )

    cv_scores_dummy = cross_val_score(
        DummyClassifier(strategy="most_frequent"),
        X,
        y,
        groups=groups,
        cv=LeaveOneGroupOut(),
    )

    avg_score = np.mean(cv_scores_svc)
    chance_level = np.mean(cv_scores_dummy)

    print(f"Average decoding accuracy: {avg_score:.2f}")
    print(f"Chance level: {chance_level:.2f}")

    return avg_score, chance_level, cv_scores_svc, pearson_corrs
