
import random
import pandas as pd
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report

from deep_learning.metrics import confusion_matrix_visualization
from deep_learning.resnet_50_model import ResNet50Custom
from settings import DELIVERABLE_DIR_TASK3, DATASET_PATH
from utils.visualization_functions import visualize_image_array
from utils.deep_learning_functions import tensor_to_image



def main(): # pylint: disable=[too-many-locals]

    # Step 1
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    dataset_unormalized = ImageFolder(root=DATASET_PATH, transform=transform)

    # visualize a random image from the dataset
    sample_image, i_sample = dataset_unormalized[
        random.randint(0, len(dataset_unormalized))
    ]
    array_visualization = tensor_to_image(sample_image)
    visualize_image_array(
        array_visualization, title=dataset_unormalized.classes[i_sample]
    )

    # Step 2
    resnet_50_custom = ResNet50Custom(num_classes=len(dataset_unormalized.classes))

    # Step 3
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = ImageFolder(root=DATASET_PATH, transform=transform)

    # Step 4
    result = resnet_50_custom.predict_on_image_folder_dataset(dataset)
    predicted_labels = result["y_pred"]
    actual_labels = result["y_true"]

    # we visualize a few images to see how the model performs
    number_images_to_visualize = 5
    for _ in range(number_images_to_visualize):
        i_random = random.randint(0, len(dataset))
        sample_image, i_label = dataset_unormalized[i_random]
        array_visualization = tensor_to_image(sample_image)
        i_class_predicted = predicted_labels[i_random]
        title = (
            f"Predicted: {dataset.classes[i_class_predicted]},"
            f" actual class: {dataset.classes[i_label]}"
        )
        visualize_image_array(array_visualization, title=title)

    # Step 5
    confusion_matrix_visualization(
        predictions=predicted_labels,
        ground_truth=actual_labels,
        labels=dataset.classes,
        save_path=DELIVERABLE_DIR_TASK3.joinpath("confusion_matrix.png"),
    )

    report = classification_report(
        y_pred=predicted_labels,
        y_true=actual_labels,
        target_names=dataset.classes,
        output_dict=True,
    )

    df_report = pd.DataFrame(report).transpose()
    # we remove some decimals to make the report more readable
    df_report_rounded = df_report.round(2)
    df_report_rounded.to_csv(DELIVERABLE_DIR_TASK3.joinpath("classification_report.csv"))


if __name__ == "__main__":
    main()
