import torchvision.datasets as datasets

data_dir = "/usr/project/xtmp/inaturalistdata_store"
full_train_dataset = datasets.INaturalist(
    root=data_dir,
    version="2019_train",
    download=True,
)
