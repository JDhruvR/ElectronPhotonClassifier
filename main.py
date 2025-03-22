import marimo

__generated_with = "0.11.25"
app = marimo.App(
    width="full",
    css_file="/home/dhruv/.local/share/mtheme/themes/wigwam.css",
)


@app.cell
def _():
    import marimo as mo
    from utils import load_electron_photon_dataset, create_transform, show_image
    from models import ResNet15_v1
    return (
        ResNet15_v1,
        create_transform,
        load_electron_photon_dataset,
        mo,
        show_image,
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
        electron_dataset_path, photon_dataset_path, (0.70, 0.15, 0.15), 512
    )
    return test_loader, train_loader, transform, val_loader


@app.cell
def _(show_image, train_loader):
    show_image(train_loader.dataset[3][0]), show_image(train_loader.dataset[-3][0])
    return


@app.cell
def _(ResNet15_v1):
    model = ResNet15_v1()
    model.summary()
    return (model,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
