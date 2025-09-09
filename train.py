# Import libraries
import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from argparse import ArgumentParser
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer
from PIL import Image, ImageEnhance
import random

scaler=StandardScaler()
pca=PCA(n_components=85)

# SVC Model
from sklearn.svm import SVC

# Define hyperparameter grid
svc_param_grid = {
    'C': np.logspace(-2, 1, 4),
    'kernel': ['rbf'],
    'gamma': np.logspace(-3, -1, 3)
}

# Create a grid search
svc_grid = GridSearchCV(SVC(), svc_param_grid, cv=5, scoring='accuracy', n_jobs=-1)

########################################################################################################################

# Random Forest Model
from sklearn.ensemble import RandomForestClassifier

# Define hyperparameter grid
rf_param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5,6,7],
    'min_samples_split': [15, 20, 25],
    'min_samples_leaf': [6, 8, 10]
}

# Create a grid search
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)

########################################################################################################################


# Logistic Regression Model
from sklearn.linear_model import LogisticRegression

# Define hyperparameter grid
lr_param_grid = {
    'C': [0.001, 0.01, 0.1, 1],
    'penalty': ['l2'],
    'solver': ['liblinear','newton-cholesky'],  
    'max_iter': [10000, 20000]
}

# Create a grid search
lr_grid = GridSearchCV(LogisticRegression(), lr_param_grid, cv=5, scoring='accuracy', n_jobs=-1)

#########################################################################################################################

# KNN Model
from sklearn.neighbors import KNeighborsClassifier

# Define hyperparameter grid
knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],  
    'weights': ['uniform', 'distance'],  
    'metric': ['euclidean', 'manhattan', 'minkowski']  
}

# Create a grid search
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=5, scoring='accuracy', n_jobs=-1)

#########################################################################################################################

# Data Augmentation
def augment_image(image):
    
    img = Image.fromarray(image.reshape(62, 47).astype('uint8'))
    
    # Random horizontal flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Random vertical flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # Random contrast adjustment
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.5, 1.2))
    
    # Random rotation
    img = img.rotate(random.uniform(-10, 10))  # Rotate between -10 to 10 degrees
    
    # Random brightness adjustment
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.5, 1.2))
    
    # Convert back to numpy array
    return np.array(img).flatten()


# Analyse the best model
def analyse_bestmodel(models, train_diff, train_labels):
    accuracy=0
    bestmodel=None
    for model in models:
        model.fit(train_diff, train_labels)
        predictions = model.predict(train_diff)
        accuracy_m=accuracy_score(train_labels, predictions)
        print(f'{model.best_estimator_} accuracy is {accuracy_m*100:.2f}%')
        if accuracy_m>accuracy:
            accuracy=accuracy_m
            bestmodel=model.best_estimator_
    print(f'Best model is {bestmodel} with accuracy {accuracy*100:.2f}%')
    return bestmodel


# Define a function to split and combine features
def split_and_combine(X):
    image1 = X[:, :2914]
    image2 = X[:, 2914:]
    return np.abs(image1 - image2)  

# Create a FunctionTransformer
splitter_transformer = FunctionTransformer(split_and_combine)

# Define the pipeline
def prepare_and_save_pipeline(bestmodel, save_path, scaler, pca):
    # Build a pipeline 
    pipeline = Pipeline([
        ('splitter_image', splitter_transformer),
        ('scaler', scaler),
        ('pca', pca),
        ('model', bestmodel)          
    ])
    
    # Save the pipeline to a file
    joblib.dump(pipeline, save_path)
    print(f"Pipeline with best model saved to: {save_path}")
    return pipeline


def train(train_data_file, model_file):
    train_data = joblib.load(train_data_file)
    train_images = train_data['data']
    train_labels = train_data['target']
    

    # Split data for each pair
    train_image1 = train_images[:, :2914]  
    train_image2 = train_images[:, 2914:]  

    augmented_image1 = []
    augmented_image2 = []

    for img1, img2 in zip(train_image1, train_image2):

        augmented_image1.append(augment_image(img1))  # Augment first image
        augmented_image2.append(augment_image(img2))  # Augment second image
    
    # Combine original and augmented data
    augmented_image1 = np.array(augmented_image1)
    augmented_image2 = np.array(augmented_image2)
    
    # Calculate absolute difference
    train_diff_original = np.abs(train_image1 - train_image2)
    train_diff_augmented = np.abs(augmented_image1 - augmented_image2)
    
    # Combine both datasets
    train_diff_combined = np.vstack((train_diff_original, train_diff_augmented))
    train_labels_combined = np.hstack((train_labels, train_labels))  # Duplicate labels for matching

    
    train_diff_scaled=scaler.fit_transform(train_diff_combined)
    train_diff_pca=pca.fit_transform(train_diff_scaled)

    # Train and evaluate different models
    models = [rf_grid, lr_grid, knn_grid,svc_grid]
    bestmodel=analyse_bestmodel(models, train_diff_pca, train_labels_combined)
    prepare_and_save_pipeline(bestmodel, save_path=model_file, scaler=scaler, pca=pca)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("train_data_file", type=str , default="train.joblib")
    parser.add_argument("model_file", type=str , default="model.joblib")
    
    args = parser.parse_args()
    train(args.train_data_file, args.model_file)

