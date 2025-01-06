import numpy as np
from benchmark_utils.utils import LabeledImage


def _compute_template_one_fold(
    algo,
    dict_alignment,
    dict_decoding,
    masker,
):
    # Get the list of subjects
    subject_list = list(dict_alignment.keys())

    # Get the list of images
    imgs = [dict_alignment[subject].img for subject in subject_list]

    # Align the images
    algo.fit(imgs)

    # Initialize the template
    template_data = np.zeros_like(masker.transform(dict_decoding[subject_list[0]].img))
    dict_aligned = dict()
    for i, subject in enumerate(subject_list):
        transformed_img = algo.transform(dict_decoding[subject].img, subject_index=i)
        dict_aligned[subject] = LabeledImage(
            img=transformed_img,
            y=dict_decoding[subject].y,
        )
        template_data += masker.transform(transformed_img) / len(subject_list)

    # Convert the template to a LabeledImage
    template = LabeledImage(
        img=masker.inverse_transform(template_data),
        y=dict_decoding[subject_list[0]].y,
    )

    return template, dict_aligned
