# PyJax PyTorch

This is example shows how you can use PyTorch for math and science calculations by using the Tensor type.

To set up this example, you will need to have Pixi installed. Pixi is package manager similar to Conda or UV, but works with different languages other than just Python. You can install Pixi using the following instructions.

### Linux or MacOS

```sh
curl -fsSL https://pixi.sh/install.sh | sh
```

### Windows Powershell

```sh
powershell -ExecutionPolicy Bypass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

## How to run the project

Once you have pixi installed you can run the examples included in this repo by installing the dependencies.

```sh
pixi install
```

Once you have the dependencies installed, you can start a new Pixi shell by running the following command:

```sh
pixi shell
```

## Running the interactive Python Notebooks

This project includes `ipykernel`, so you can run this inside of VS Code, but if you want to run the notebooks outside of VS Code, you will need to add Jupyterlab.

The important files in this project are as follows:

* `cleaned_data.csv` is the data we will use to train our model
* `eda.ipynb` is the notebook that was used to convert the original data into the CSV used for training
* `pytorch-demo.ipynb` is the notebook that was used to demo the presentation and train the model

Copyright 2027 David Fekke
