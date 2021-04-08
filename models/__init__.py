from models.UNet import UNet

def get_model(name, model_opts):
    if name == "unet":
        model = UNet(**model_opts)
        return model
    else:
        raise RuntimeError("model \"{}\" not available".format(name))