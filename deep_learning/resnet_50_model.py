from pathlib import Path
from typing import Optional, List, Dict

import torch
from torch import nn, Tensor
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
import torch.nn.functional as F


TASK_3_DIR = Path.cwd().parent.joinpath("TASK 3 Data")
TASK_3_DIR = Path(
    "/home/cedric/repos/sdgai_assignment/TASK 3 Data-20230913T191958Z-001"
)
WEIGHTS_PATH = TASK_3_DIR / "TASK 3 Data/model.pth"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ResNet50Custom:
    def __init__(
        self,
        num_classes: int,
        learning_rate: float = 5e-4,
        device: str = DEVICE,
        weights_path: Optional[str] = WEIGHTS_PATH,
    ):
        self.number_classes = num_classes
        self.device = device
        self.weights_path = weights_path

        if self.weights_path:
            self.model = self._load_scene_classification_model()
        else:
            self.model = self._load_pre_trained_model()

        self.learning_rate = learning_rate
        self.optimizer = self._define_optimizer(learning_rate)

        if self.device == "cpu":
            self.model = self.model.cpu()
        else:
            self.model = self.model.cuda()

    def predict_probabilities(self, tensor: Tensor) -> Tensor:

        if len(tensor.shape) != 4:
            # model needs a batch dimension otherwise it will raise an error
            tensor = tensor.unsqueeze(0)

        # data and model on different devices will cause an error
        tensor = tensor.to(self.device)

        # we use the eval mode to avoid updating the gradients
        self.model.eval()

        return F.softmax(self.model(tensor), dim=1)

    def _load_pre_trained_model(self):
        model = resnet50(pretrained=True).float()

        # Update the final fully connected layer for our number of classes
        model.fc = nn.Linear(model.fc.in_features, self.number_classes)

        # we initialize only the last layer
        nn.init.xavier_uniform_(model.fc.weight)

        return model

    def _load_scene_classification_model(self):
        model = resnet50(pretrained=True).float()

        # Update the final fully connected layer for our number of classes
        model.fc = nn.Linear(model.fc.in_features, self.number_classes)

        # pre trained weights expect 10 classes as output
        model.load_state_dict(torch.load(self.weights_path, map_location=self.device))

        return model

    def predict_on_image_folder_dataset(
        self, image_dataset: ImageFolder
    ) -> Dict[str, List[int]]:
        predicted_labels = []
        actual_labels = []
        for image, i_label in image_dataset:
            predicted_probabilities = self.predict_probabilities(image)
            i_class_predicted = predicted_probabilities.argmax().item()

            predicted_labels.append(i_class_predicted)
            actual_labels.append(i_label)
        return {"y_pred": predicted_labels, "y_true": actual_labels}

    def _define_optimizer(self, learning_rate):
        # Extracting non-final layer parameters
        params_pre_trained = [
            param
            for name, param in self.model.named_parameters()
            if name not in ["fc.weight", "fc.bias"]
        ]

        # Defining optimizer with differential learning rates
        # final layer has a more aggressive learning rate
        optimizer = torch.optim.Adam(
            [
                {"params": params_pre_trained},
                {
                    "params": self.model.fc.parameters(),
                    "lr": learning_rate * 10,
                },  # final layer
            ],
            lr=self.learning_rate,
            weight_decay=0.001,
        )

        return optimizer
