import torch

from deep_learning.resnet_50_model import ResNet50Custom


def test_ResNet50Custom_number_classes_out_is_correct():
    resnet_50_custom = ResNet50Custom(num_classes=10, weights_path=None, device="cpu")
    assert resnet_50_custom.model.fc.out_features == 10


def test_ResNet50Custom_optimizer_has_different_learning_rates():
    resnet_50_custom = ResNet50Custom(num_classes=10, weights_path=None, device="cpu")

    assert (
        len(resnet_50_custom.optimizer.param_groups) == 2
    ), "There should be two parameter groups, one for the final layer and one for the rest of the model"
    assert (
        resnet_50_custom.optimizer.param_groups[0]["lr"]
        == resnet_50_custom.learning_rate
    )
    assert (
        resnet_50_custom.optimizer.param_groups[1]["lr"]
        == resnet_50_custom.learning_rate * 10
    ), "The learning rate for the final layer should be 10 times the learning rate of the rest of the model"


def test_ResNet50Custom_model_is_on_cpu():
    resnet_50_custom = ResNet50Custom(num_classes=10, weights_path=None, device="cpu")
    expected = torch.device("cpu")
    # checking the device on the whole model
    layers = list(resnet_50_custom.model.parameters())
    for i_layer, layer in enumerate(layers):
        assert (
            layer.device == expected
        ), f"The model should be on the cpu. layer {i_layer} is on {layer.device}"


def test_ResNet50Custom_model_is_on_cuda():
    resnet_50_custom = ResNet50Custom(num_classes=10, weights_path=None, device="cuda")
    expected = torch.device("cuda:0")
    # checking the device on the whole model
    layers = list(resnet_50_custom.model.parameters())
    for i_layer, layer in enumerate(layers):
        assert (
            layer.device == expected
        ), f"The model should be on cuda. layer {i_layer} is on {layer.device}"


def test_model_forward_returns_correct_output():
    resnet_50_custom = ResNet50Custom(num_classes=10, weights_path=None, device="cuda")

    array = torch.rand(1, 3, 224, 224).cuda()

    output = resnet_50_custom.model(array)

    assert output.shape == (1, 10)
