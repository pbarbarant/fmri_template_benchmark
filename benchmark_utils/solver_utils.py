import numpy as np
from benchmark_utils.datasets_utils import LabeledImage


def compute_template(
    algo,
    dataset,
):
    # Get the list of subjects
    subject_list = list(dataset.dict_alignment.keys())

    # Get the list of images
    imgs = [dataset.dict_alignment[subject] for subject in subject_list]

    # Align the images
    algo.fit(imgs)

    # Initialize the template
    template_data = np.zeros_like(
        dataset.masker.transform(dataset.dict_decoding[subject_list[0]].img)
    )
    dict_aligned = dict()
    for i, subject in enumerate(subject_list):
        transformed_img = algo.transform(
            dataset.dict_decoding[subject].img, subject_index=i
        )
        dict_aligned[subject] = LabeledImage(
            img=transformed_img,
            y=dataset.dict_decoding[subject].y,
        )
        template_data += dataset.masker.transform(transformed_img) / len(
            subject_list
        )

    # Convert the template to a LabeledImage
    template = LabeledImage(
        img=dataset.masker.inverse_transform(template_data),
        y=dataset.dict_decoding[subject_list[0]].y,
    )

    dataset.template = template
    dataset.dict_aligned = dict_aligned

    return dataset
