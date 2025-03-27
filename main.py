import marimo

__generated_with = "0.11.25"
app = marimo.App(
    width="full",
    css_file="/home/dhruv/.local/share/mtheme/themes/wigwam.css",
)


@app.cell
def _():
    import marimo as mo

    import torch.optim as optim
    import torch.nn as nn

    from utils import load_electron_photon_dataset, create_transform, show_image
    from models import ResNet15_v1, ResNet15_v2
    from train import train, evaluate_model
    return (
        ResNet15_v1,
        ResNet15_v2,
        create_transform,
        evaluate_model,
        load_electron_photon_dataset,
        mo,
        nn,
        optim,
        show_image,
        train,
    )


@app.cell
def _():
    electron_dataset_path = "data/SingleElectron249k.hdf5"
    photon_dataset_path = "data/SinglePhoton249k.hdf5"
    return electron_dataset_path, photon_dataset_path


@app.cell
def _(
    electron_dataset_path,
    load_electron_photon_dataset,
    photon_dataset_path,
):
    train_loader, val_loader, test_loader, transform = load_electron_photon_dataset(
        electron_dataset_path, photon_dataset_path, (0.01, 0.01, 0.98), 512
    )
    return test_loader, train_loader, transform, val_loader


@app.cell
def _(show_image, train_loader):
    show_image(train_loader.dataset[3][0]), show_image(train_loader.dataset[-3][0])
    return


@app.cell
def _(ResNet15_v2):
    model = ResNet15_v2()
    model.summary(batch_size=512)
    return (model,)


@app.cell
def _(model, nn, train, train_loader, val_loader):
    train(
        model=model,
        epochs=5,
        criterion=nn.CrossEntropyLoss(),
        train_loader=train_loader,
        val_loader=val_loader,
        lr=1e-3,
        device='cpu',
        scheduler_type='plateau'
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
