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
git commit -m "Add PlantVillage dataset to DVC" # Commit the changes to Git
git push # Push the changes to the remote repository
```

This command will create the following files in the project root:

```
├── PlantVillage.dvc # DVC file that tracks the PlantVillage directory
├──.dvc/
    ├── config
    ├── .gitignore
    ├── cache/
        ├── md5/
        ├── ... # md5 hashes of the files in the PlantVillage directory
```

## Add remote to the DVC repository

To store the data remotely, we need to add a remote to the DVC repository. In this example, we will use Google Drive as the remote storage. Run the following command in the terminal:

```bash
dvc remote add -d myremote /tmp # In this case we use the Unix temporary directory as the remote storage
dvc push # Push the data to the remote storage
```

## Tracking changes to the data

DVC allows us to track changes to the data over time. If we make any changes to the data, we can run the following command to update the DVC tracking:

```bash
dvc add PlantVillage # Update the DVC tracking for the PlantVillage directory
git commit -m "Update PlantVillage dataset" # Commit the changes to Git
git push # Push the changes to the remote repository
dvc push # Push the updated data to the remote storage
```

## Pulling data from the remote storage

If we want to pull the data from the remote storage to a new machine, we can run the following command in the terminal:

```bash
git clone <repository-url> # Clone the repository to the new machine
cd <repository-name> # Change to the project directory
dvc pull # Pull the data from the remote storage
```

This command will download the data from the remote storage and place it in the `PlantVillage` directory in the project root.

