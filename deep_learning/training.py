from torch import nn


def training(model: nn.Module, data_loader, criterion, optimizer, num_epochs: int):

    if not issubclass(type(model), nn.Module):
        if hasattr(model, "model") and issubclass(type(model.model), nn.Module):
            model = model.model
        else:
            raise ValueError("model should be a pytorch model")

    for epoch in range(num_epochs):

        model.train()  # Set the model to training mode

        running_loss = 0.0
        for i, (inputs, labels) in enumerate(data_loader):
            # Move inputs and labels to the target device (GPU or CPU)
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        # Optionally, apply the learning rate scheduler
        scheduler.step()

        epoch_loss = running_loss / len(data_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    print("Training complete!")


torch.save(model.state_dict(), "fine_tuned_model.pth")
