# Plant Village Classification

This project aims to use different tools available in the field of machine learning and neural network training to build and deploy a model to classify plant diseases. The task and the dataset itself is not the focus of this project. The particular dataset was chosen because:

* It is a well-known dataset in the computer vision community.
* It is a fairly large dataset, which allows us to train a reasonably good model.
* The dataset is simple enough to be used as a tutorial for building and deploying a machine learning model.

## Project demo

Try the quick demo on [Render](https://plant-village-kxps.onrender.com/).

## Goals of this project

The main goals of this project are:

1. Familiarize with data version control using DVC.
2. Build a neural network model using PyTorch to classify plant diseases.
3. Track experiements using Weights & Biases.
4. Use ONNX to export the trained model and deploy it using Flask

## Getting Started

### Preparing a repository

This entire project will be built in a Git repository. Create an empty repository on GitHub. Don't forget to initialize `.gitignore` for python files. Now clone the repository to your local machine:

```bash
git clone https://github.com/asprasan/PlantVillageClassification.git
cd PlantVillageClassification
```

### Setting up the environment

Here, we use conda to manage our environment. You can use any other tool you are comfortable with. Create a new conda environment and activate it:

```bash
conda create -n plant python=3.10
conda activate plant
```

Let's install some of the dependencies we will need for this project:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install dvc wandb
```

