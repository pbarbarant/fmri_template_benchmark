import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import (
    LeaveOneGroupOut,
    cross_val_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from nilearn.maskers import NiftiLabelsMasker
from nilearn import image


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


def compute_pearson_corrs(dataset):
    template_img = dataset.template.img
    clustering_img = dataset.clustering_img
    masker = dataset.masker
    labels_masker = NiftiLabelsMasker(
        labels_img=clustering_img, mask_img=masker.mask_img_
    ).fit()
    pearson_corrs = []
    for subject in dataset.subjects:
        subject_img = dataset.dict_aligned[subject].img
        subject_corr = pearson_corr_parcels(subject_img, template_img, labels_masker)
        pearson_corrs.append(subject_corr)
    return pearson_corrs


def pearson_corr_parcels(img1, img2, labels_masker):
    """Compute the Pearson correlation between two images
    by averaging the signal in each parcel."""
    data1 = labels_masker.transform(img1)
    data2 = labels_masker.transform(img2)
    n_parcels = data1.shape[1]
    return np.mean(
        [np.corrcoef(data1[:, i], data2[:, i])[0, 1] for i in range(n_parcels)]
    )


def evaluate_task_dataset(dataset, max_iter=100):
    dict_aligned = dataset.dict_aligned
    masker = dataset.masker

    # Compute the voxel-wise pearson correlation between all subjects
    # and the template
    pearson_corrs = compute_pearson_corrs(dataset)

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
        n_jobs=-1,
    )

    cv_scores_dummy = cross_val_score(
        DummyClassifier(strategy="most_frequent"),
        X,
        y,
        groups=groups,
        cv=LeaveOneGroupOut(),
        n_jobs=-1,
    )

    avg_score = np.mean(cv_scores_svc)
    chance_level = np.mean(cv_scores_dummy)

    print(f"Average decoding accuracy: {avg_score:.2f}")
    print(f"Chance level: {chance_level:.2f}")

    return avg_score, chance_level, cv_scores_svc, pearson_corrs


def classify_subject_movie(template_img, img, y, labels_masker):
    segments_ids = np.unique(y)
    res = []
    # Get the list of segments for the template
    segments_template = []
    segments_subject = []
    for segment_id in segments_ids:
        # Get the slice of indices corresponding to the segment
        segment_slice = y == segment_id
        # Get the image segment for the template
        segments_template.append(image.index_img(template_img, segment_slice))
        # Get the image segment for the subject
        segments_subject.append(image.index_img(img, segment_slice))

    for i in range(len(segments_ids)):
        # For each movie segment of the subject
        segment_subject = segments_subject[i]
        correct_id = segments_ids[i]
        corr_list = []
        for j in range(len(segment_id)):
            segment_template = segments_template[j]
            # Compute the Pearson correlation with each segment of the template
            corr_list.append(
                pearson_corr_parcels(segment_template, segment_subject, labels_masker)
            )
        # Predict the segment with the highest correlation
        predicted_id = segments_ids[np.argmax(corr_list)]
        res.append(predicted_id == correct_id)
    return np.mean(res)


def evaluate_movie_dataset(dataset):
    dict_aligned = dataset.dict_aligned
    masker = dataset.masker

    # Compute the voxel-wise pearson correlation between all subjects
    # and the template
    pearson_corrs = compute_pearson_corrs(dataset)

    # Create cross-validation object on each subject
    X, y = compute_X_y(dict_aligned, masker)
    groups = compute_groups(dict_aligned)

    labels_masker = NiftiLabelsMasker(
        labels_img=dataset.clustering_img, mask_img=masker.mask_img_
    ).fit()
    cv_scores_classif = []
    for subject in dataset.subjects:
        print(f"Subject {subject}")
        cv_scores_classif.append(
            classify_subject_movie(
                template_img=dataset.template.img,
                img=dict_aligned[subject].img,
                y=dict_aligned[subject].y,
                labels_masker=labels_masker,
            )
        )

    cv_scores_dummy = cross_val_score(
        DummyClassifier(strategy="most_frequent"),
        X,
        y,
        groups=groups,
        cv=LeaveOneGroupOut(),
        n_jobs=-1,
    )

    avg_score = np.mean(cv_scores_classif)
    chance_level = np.mean(cv_scores_dummy)

    print(f"Average decoding accuracy: {avg_score:.2f}")
    print(f"Chance level: {chance_level:.2f}")

    return avg_score, chance_level, cv_scores_classif, pearson_corrs


def evaluate_dataset(dataset, max_iter=100):
    if True:
        return evaluate_movie_dataset(dataset)
    # if dataset.name.lower().startswith("budapest"):
    #     return evaluate_movie_dataset(dataset)
    # else:
    #     return evaluate_task_dataset(dataset, max_iter=max_iter)
