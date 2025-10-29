# Jellyfish classification project using transfer learning 🪼


This project implements a classification model using transfer learning from pre-trained models, with Tensorflow/Keras. The approach involves:
- Fetching and preparing a dataset from the iNaturalist public API
- Loading a pre-trained base model
- Adding a classification head
- Training in two stages: feature extraction and fine-tuning


## Installation


1. Clone the repository:
```bash
git clone https://github.com/elisectr/jellyfish-classification.git
cd jellyfish_classif
```

2. Build the docker image :
```bash
docker build -t jellyfish-classif .
```

3. Run with Docker Compose :
```bash
docker compose up
```
The config file (`config.yaml`) defines all parameters for downloading the dataset, creating the model, training ...


## Dataset

The dataset is fetched from iNaturalist API, downloading images from 5 different species, as defined in the yaml config file:

    - Aurelia aurita
    - Pelagia noctiluca
    - Chrysaora hysoscella
    - Cotylorhiza tuberculata
    - Rhizostoma pulmo


## Model
- **Architecture:** [ResNet50/ResNet101/MobileNetV2/VGG16/InceptionV3]
- **Pre-trained weights:** ImageNet
- **Classification head** as follows:
```
Input -> Base Model -> GlobalAveragePooling2D -> Dropout(0.5) -> Dense(256, softmax)
```

## Notebooks
Jupyter notebooks for data exploration and experimentation are available in the `notebooks/` directory.

