import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import albumentations as A

np.random.seed(42)
tf.random.set_seed(42)

BASE_PATH = r"C:\Users\Aman\Desktop\DeepLearning\cassava-leaf-disease-classification (2)\cassava-leaf-disease-classification"
TRAIN_IMAGES_PATH = os.path.join(BASE_PATH, "train_images")
TRAIN_CSV_PATH = os.path.join(BASE_PATH, "train.csv")
LABEL_MAP_PATH = os.path.join(BASE_PATH, "label_num_to_disease_map.json")

IMG_SIZE = 512
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2

with open(LABEL_MAP_PATH, 'r') as f:
    label_map = json.load(f)
    # Convert keys to integers
    label_map = {int(k): v for k, v in label_map.items()}

for label, disease in label_map.items():
    print(f"  {label}: {disease}")

train_df = pd.read_csv(TRAIN_CSV_PATH)
print(f"\nDataset Size: {len(train_df)} images")
print(f"\nFirst few rows:")
print(train_df.head())

print("\nClass Distribution:")
class_counts = train_df['label'].value_counts().sort_index()
for label, count in class_counts.items():
    print(f"  {label_map[label]}: {count} images ({count/len(train_df)*100:.2f}%)")

plt.figure(figsize=(12, 6))
sns.countplot(data=train_df, x='label')
plt.title('Distribution of Disease Classes')
plt.xlabel('Disease Label')
plt.ylabel('Count')
plt.xticks(range(len(label_map)), [label_map[i] for i in range(len(label_map))], rotation=45, ha='right')
plt.tight_layout()
plt.show()

def get_train_augmentation():
    """
    Define training augmentation pipeline using Albumentations.
    Includes various transformations to improve model generalization.
    """
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Transpose(p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.GaussianBlur(),
            A.MotionBlur(),
        ], p=0.3),
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.3),
        ], p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_valid_augmentation():
    """
    Define validation augmentation pipeline.
    Only includes resizing and normalization.
    """
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

train_transform = get_train_augmentation()
valid_transform = get_valid_augmentation()

class CassavaDataGenerator(keras.utils.Sequence):
    """
    Custom data generator for loading and augmenting cassava leaf images.
    """
    def __init__(self, dataframe, img_dir, augmentation=None, batch_size=32, shuffle=True):
        self.dataframe = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.augmentation = augmentation
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_classes = len(label_map)
        self.on_epoch_end()
    
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.dataframe) / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data"""
        # Get batch indices
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.dataframe))
        batch_indices = self.indices[start_idx:end_idx]
        
        # Generate data
        X, y = self.__data_generation(batch_indices)
        return X, y
    
    def on_epoch_end(self):
        """Shuffle indices after each epoch"""
        self.indices = np.arange(len(self.dataframe))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __data_generation(self, batch_indices):
        """Generate batch of augmented images and labels"""
        X = np.empty((len(batch_indices), IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        y = np.empty((len(batch_indices)), dtype=np.int32)
        
        for i, idx in enumerate(batch_indices):
            # Load image
            img_name = self.dataframe.loc[idx, 'image_id']
            img_path = os.path.join(self.img_dir, img_name)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply augmentation
            if self.augmentation:
                augmented = self.augmentation(image=image)
                image = augmented['image']
            
            X[i,] = image
            y[i] = self.dataframe.loc[idx, 'label']
        
        # Convert labels to categorical
        y = keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, y

train_data, val_data = train_test_split(
    train_df, 
    test_size=VAL_SPLIT, 
    stratify=train_df['label'], 
    random_state=42
)

train_generator = CassavaDataGenerator(
    train_data,
    TRAIN_IMAGES_PATH,
    augmentation=train_transform,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_generator = CassavaDataGenerator(
    val_data,
    TRAIN_IMAGES_PATH,
    augmentation=valid_transform,
    batch_size=BATCH_SIZE,
    shuffle=False
)

def visualize_augmentations(generator, num_samples=8):
    """Visualize augmented images from the generator"""
    X_batch, y_batch = generator[0]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(X_batch))):
        img = X_batch[i]
        # Denormalize for visualization
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        label_idx = np.argmax(y_batch[i])
        disease_name = label_map[label_idx]
        
        axes[i].imshow(img)
        axes[i].set_title(f'{disease_name}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

    visualize_augmentations(train_generator)

    def build_model(num_classes=5, img_size=IMG_SIZE):
   
    # Load pre-trained EfficientNetB3
    base_model = EfficientNetB3(
        include_top=False,
        weights='imagenet',
        input_shape=(img_size, img_size, 3)
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Build model
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model

model = build_model(num_classes=len(label_map))

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
os.makedirs('models', exist_ok=True)

checkpoint = ModelCheckpoint(
    'models/best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True,
    verbose=1
)

callbacks = [checkpoint, reduce_lr, early_stopping]

history_phase1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS // 2,
    callbacks=callbacks,
    verbose=1
)

base_model = model.layers[1]
base_model.trainable = True

for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_phase2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS // 2,
    callbacks=callbacks,
    verbose=1
)

model.save('models/final_model.keras')