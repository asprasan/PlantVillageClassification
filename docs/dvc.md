# Data Version Control with DVC

Download the Plant Village classification dataset from Kaggle. Extract the downloaded zip file and place the extracted folder in the root directory of this project. The extracted folder should be named `PlantVillage`.

The data would look something like:

```
PlantVillage
├── Potato___Early_blight
├── Potato___Late_blight
├── Potato___healthy
├── Tomato___Bacterial_spot
├── ...
├── 
```

In this project, we will only use 2 classes of the dataset for simplicity. Remove all directories except `Pepper__bell___Bacterial_spot` and `Pepper__bell___healthy`. The final structure of the data should look like:

```
PlantVillage
├── Pepper__bell___Bacterial_spot
├── Pepper__bell___healthy
```

## Initialize DVC in the project directory

To track the data with DVC, we need to initialize DVC in the project directory. Run the following command in the terminal:

```bash
dvc init # Initialize DVC in the project directory and stages the changes to Git
git commit -m "Initialize DVC" # Commit the changes to Git
```

This command should create a `.dvc` directory in the project root, which contains the DVC configuration files.

```
.dvc
├── config
├── .gitignore
```

## Add the data to DVC

To add the data to DVC, run the following command in the terminal:

```bash
dvc add PlantVillage # Add the data to DVC
```