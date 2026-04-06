# CVRIE - Medical Artificial Intelligence

## Project Overview
This project was developed as part of the **G-AIA-400** module at **Epitech**. The objective is to enhance the diagnostic capabilities of CVRIE, a medical AI, through two distinct machine learning approaches: unsupervised learning for data discovery and supervised learning for specialized medical diagnosis.

The project is divided into two core components:
1. **Testimony Analysis (Unsupervised)**: Clustering and segmentation of raw patient data.
2. **Hand Fracture Classification (Supervised)**: Detection and classification of specific hand traumas using a dedicated dataset.


## Part 1: Unsupervised Learning - Testimony Clustering
This section focuses on processing raw, unlabeled medical testimonies to identify recurring themes and patient groups.

### Technical Pipeline:
* **Preprocessing**: Text normalization, stop-word removal, and TF-IDF vectorization.
* **Dimensionality Reduction**: Implementation of **PCA** (Principal Component Analysis) and **t-SNE** for high-dimensional data visualization.
* **Clustering**: Application of the **K-means** algorithm.
* **Evaluation**: Models are assessed using the Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Score.

---

## Part 2: Supervised Learning - Hand Fractures
For the supervised learning task, we chose to specialize the model in the detection and classification of **hand fractures**. 

### Objectives:
* Train a classification model to distinguish between different types of hand bone injuries.
* Optimize model performance through hyperparameter tuning.
* Validation using confusion matrices, precision, recall, and F1-score metrics.

---

## Technologies Used
* **Language**: Python 3.x
* **Data Processing**: Pandas, Numpy
* **Visualization**: Matplotlib, Seaborn
* **Machine Learning**: Scikit-learn
* **Environment**: Jupyter Lab / Notebook

---

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/cvrie-mirror.git
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib jupyter
   ```

3. Launch the analysis environment:
   ```bash
   jupyter lab
   ```

---
