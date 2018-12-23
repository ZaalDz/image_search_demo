from torchvision import transforms


def get_transformation():
    """
    transformation for input image
    Returns:

    """
    return transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def extract_features(model, layer, image):
    preprocess_image = get_transformation()
    transformed_image = preprocess_image(image).unsqueeze(0)
    features = None

    def get_data(m, i, o):
        nonlocal features
        features = o

    hook = layer.register_forward_hook(get_data)
    model(transformed_image).detach().numpy()
    hook.remove()
    return features.detach().numpy()

