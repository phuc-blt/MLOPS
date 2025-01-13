import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import mlflow
import mlflow.tensorflow

# Check GPU availability
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to avoid allocation of all GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is available and memory growth is enabled.")
    except RuntimeError as e:
        print(e)
else:
    print("GPU is not available. Using CPU.")

def prepare_data(data_path):
    """Prepare the dataset from the given path."""
    classes, file_paths = zip(*[
        (label, os.path.join(data_path, label, image))
        for label in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, label))
        for image in os.listdir(os.path.join(data_path, label))
    ])
    return pd.DataFrame({'FilePath': file_paths, 'Class': classes})

def build_model(input_shape, num_classes):
    """Build and compile a CNN model."""
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(data_path, model_path, mlflow_experiment):
    """Train the model and log to MLflow."""
    # Prepare data
    df = prepare_data(data_path)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['Class'], random_state=42)

    train_gen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2,
                                   height_shift_range=0.2, horizontal_flip=True)
    val_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_dataframe(
        train_df, x_col='FilePath', y_col='Class',
        target_size=(224, 224), batch_size=32, class_mode='categorical'
    )
    val_data = val_gen.flow_from_dataframe(
        val_df, x_col='FilePath', y_col='Class',
        target_size=(224, 224), batch_size=32, class_mode='categorical'
    )

    # Build model
    input_shape = (224, 224, 3)
    num_classes = len(train_data.class_indices)
    model = build_model(input_shape, num_classes)

    # Log to MLflow
    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run():
        mlflow.tensorflow.autolog()

        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=10,
            steps_per_epoch=len(train_data),
            validation_steps=len(val_data)
        )

        # Save model
        model.save(model_path)

if __name__ == "__main__":
    train_data_path = "/home/phuc/mlops/data/train"
    model_save_path = "./saved_model/brain_tumor_model.h5"
    experiment_name = "Brain_Tumor_Detection"

    train_model(train_data_path, model_save_path, experiment_name)
