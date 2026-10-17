# 🌿 Plant Disease Detection Using Image Classification

A deep learning project for detecting plant diseases from leaf images using YOLOv8 and image processing techniques. The project aims to support early disease identification and precision agriculture through an easy-to-use web application.

## Project Overview

Plant diseases can significantly affect crop health, agricultural productivity, and food security. Manual disease identification can be time-consuming and may require expert knowledge.

This project uses deep learning and computer vision to analyze plant leaf images and identify disease classes. The system includes image preprocessing, model training, performance evaluation, and image-based prediction.

## Key Features

- Plant leaf image analysis using deep learning.
- Image preprocessing and data cleaning.
- Image resizing, normalization, and augmentation.
- YOLOv8-based model training and prediction.
- Detection results with annotated images.
- Streamlit-based web application.
- Support for early plant disease identification.

## Technologies Used

- **Python** — programming and model development
- **YOLOv8 (Ultralytics)** — deep learning model
- **PyTorch** — deep learning framework
- **OpenCV** — image processing
- **Pillow (PIL)** — image handling and preprocessing
- **NumPy** — numerical operations
- **Streamlit** — web application interface
- **Google Colab** — model training environment
- **Jupyter Notebook** — experimentation and implementation

## Dataset

The project presentation identifies PlantVillage and Kaggle as data sources for plant leaf images.

- PlantVillage: https://plantvillage.psu.edu/
- Kaggle: https://www.kaggle.com/datasets

The dataset contains plant leaf images used for training and evaluating the disease detection model.

**Note:** The exact Kaggle dataset URL and final dataset size should be added once confirmed from the dataset used in the project.

## Project Methodology

### 1. Data Collection
Collect labeled plant leaf images from plant disease datasets.

### 2. Image Preprocessing
- Resize images to 416 × 416 pixels.
- Convert images to RGB format.
- Remove or skip corrupted images.
- Apply image augmentation to improve model generalization.
- Validate YOLO-format labels.

### 3. Dataset Splitting
Divide the dataset into training, validation, and test sets to support model training and evaluation.

### 4. Model Training
Use the YOLOv8 architecture to train the model on the prepared dataset. Training parameters include image size, batch size, epochs, and early stopping.

### 5. Model Evaluation
Evaluate model performance using classification accuracy, precision, recall, F1-score, and mean Average Precision (mAP50), as applicable to the model's prediction task.

### 6. Image Prediction
Upload a plant leaf image and use the trained model to generate predictions and an annotated output image.

### 7. Web Application
Provide a user-friendly interface for image-based prediction using Streamlit.

## Model Performance

The project presentation reports the following results:

| Evaluation Metric | Reported Result |
|---|---:|
| Classification Accuracy | 96% |
| Precision | 94.2% |
| Recall | 93.8% |
| F1-Score | 94.0% |
| mAP50 | 91.5% |
| Disease Classes | 38 |

These values are the results reported in the project presentation. Actual performance may vary depending on the dataset, test split, and evaluation method.

## How to Run the Project

### 1. Clone or Download the Repository

Download the repository from GitHub using **Code → Download ZIP**, then extract the files.

Alternatively, clone it using Git:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Plant-Disease-Detection
```

### 2. Install Dependencies

Install the required Python libraries:

```bash
pip install ultralytics streamlit opencv-python pillow numpy pyyaml
```

Additional libraries may be required depending on the imports used in the notebook or application.

### 3. Prepare the Dataset and Model

- Download and organize the plant disease dataset.
- Prepare the dataset configuration file required by YOLOv8.
- Train the model or provide the trained model weights.
- Ensure the trained `best.pt` file is available at the path expected by the application.

### 4. Run the Notebook

Open the Jupyter Notebook in Jupyter Notebook or Google Colab and execute the cells in the required order.

### 5. Run the Web Application

If the repository contains the Streamlit application, run:

```bash
streamlit run app.py
```

Replace `app.py` with the actual application filename if it is different.

**Note:** The exact run command and required files depend on the final application code included in the repository.

## Applications

- Early identification of plant diseases.
- Agricultural crop monitoring.
- Precision agriculture.
- Smart farming applications.
- Decision support for farmers.
- Computer vision and deep learning research.

## Future Improvements

- Add support for more plant species and disease classes.
- Improve performance on real-world images captured in different lighting conditions.
- Display disease confidence scores and additional information.
- Provide disease prevention and treatment guidance.
- Optimize the model for mobile and edge devices.
- Improve deployment and real-time prediction capabilities.

## Live Demo

Try the deployed application:

https://plant-disease-app-final-ppjpw5azn28mst3hdyeest.streamlit.app/

## Project Team

- Abhinav Pradhan
- Mohit Anand
- Abhishek Singh

## References

- PlantVillage: https://plantvillage.psu.edu/
- Ultralytics YOLO: https://docs.ultralytics.com/
- OpenCV: https://opencv.org/
- PyTorch: https://pytorch.org/
- Streamlit: https://streamlit.io/

## Conclusion

This project demonstrates how deep learning and computer vision can be applied to plant disease identification. By combining image preprocessing, YOLOv8, model evaluation, and a Streamlit interface, the system provides a foundation for accessible and scalable plant disease detection in agriculture.
