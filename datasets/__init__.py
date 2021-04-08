from datasets.DSB2018Dataset import DSB2018Dataset

def get_dataset(name, dataset_opts):
    if name=="dsb2018":
        return DSB2018Dataset(**dataset_opts)
    else:
        raise RuntimeError("Dataset {} not available".format(name))

