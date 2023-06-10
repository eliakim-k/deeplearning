import torch

# Check if oneDNN is available as the device backend for PyTorch
if not torch.backends.mkldnn.is_available():
    print("oneDNN is not available. Please check your installation.")
else:
    # Use Intel Iris Xe GPU as the device for computation
    device = torch.device("opencl")

    """# Specify the platform and device ID of the Intel Iris Xe GPU
    platform_id = 0
    device_id = 0

    # Set the device to the Intel Iris Xe GPU
    device = torch.device("opencl:{}".format(device_id))"""

    # Define tensor on the device
    x = torch.randn(3, 3, device=device)

    # Define a PyTorch neural network model
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2),
        torch.nn.Softmax(dim=1)
    )

    # Move the model to the device
    model.to(device)

    # Compute the output of the model on the input tensor
    output = model(x)

    # Print the output
    print(output)