import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import albumentations as A

np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_PATH = r"C:\Users\Prasukh\Desktop\DeepLearning\cassava-leaf-disease-classification (2)\cassava-leaf-disease-classification"
TRAIN_IMAGES_PATH = os.path.join(BASE_PATH, "train_images")
TRAIN_CSV_PATH = os.path.join(BASE_PATH, "train.csv")
LABEL_MAP_PATH = os.path.join(BASE_PATH, "label_num_to_disease_map.json")

IMG_SIZE = 224  # Smaller size for custom CNN
BATCH_SIZE = 32
EPOCHS = 50
INITIAL_LEARNING_RATE = 1e-3
VAL_SPLIT = 0.2

# ============================================================================
# LOAD LABEL MAP AND DATA
# ============================================================================
with open(LABEL_MAP_PATH, 'r') as f:
    label_map = json.load(f)
    label_map = {int(k): v for k, v in label_map.items()}

train_df = pd.read_csv(TRAIN_CSV_PATH)
print(f"Dataset Size: {len(train_df)} images")
print(f"Number of classes: {len(label_map)}")
print("\nClass Distribution:")
for label, disease in label_map.items():
    count = len(train_df[train_df['label'] == label])
    print(f"  {label}: {disease} - {count} images")

# ============================================================================
# DATA AUGMENTATION
# ============================================================================
def get_train_augmentation():
    """Training augmentation pipeline"""
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Transpose(p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.GaussianBlur(),
            A.MotionBlur(),
        ], p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.5),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.CoarseDropout(max_holes=8, max_height=24, max_width=24, p=0.3),
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_valid_augmentation():
    """Validation augmentation pipeline"""
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# ============================================================================
# DATA GENERATOR
# ============================================================================
class CassavaDataGenerator(keras.utils.Sequence):
    """Custom data generator for loading and augmenting cassava leaf images"""
    def __init__(self, dataframe, img_dir, augmentation=None, batch_size=32, shuffle=True):
        self.dataframe = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.augmentation = augmentation
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_classes = len(label_map)
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))
    
    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.dataframe))
        batch_indices = self.indices[start_idx:end_idx]
        X, y = self.__data_generation(batch_indices)
        return X, y
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.dataframe))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __data_generation(self, batch_indices):
        X = np.empty((len(batch_indices), IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        y = np.empty((len(batch_indices)), dtype=np.int32)
        
        for i, idx in enumerate(batch_indices):
            img_name = self.dataframe.loc[idx, 'image_id']
            img_path = os.path.join(self.img_dir, img_name)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.augmentation:
                augmented = self.augmentation(image=image)
                image = augmented['image']
            
            X[i,] = image
            y[i] = self.dataframe.loc[idx, 'label']
        
        y = keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, y

# ============================================================================
# PREPARE DATA SPLITS
# ============================================================================
train_data, val_data = train_test_split(
    train_df, 
    test_size=VAL_SPLIT, 
    stratify=train_df['label'], 
    random_state=42
)

train_generator = CassavaDataGenerator(
    train_data,
    TRAIN_IMAGES_PATH,
    augmentation=get_train_augmentation(),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_generator = CassavaDataGenerator(
    val_data,
    TRAIN_IMAGES_PATH,
    augmentation=get_valid_augmentation(),
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"\nTraining samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")

# ============================================================================
# CUSTOM CNN-DNN ARCHITECTURE
# ============================================================================
def build_custom_cnn(input_shape=(224, 224, 3), num_classes=5):
    """
    Custom CNN-DNN Architecture with multiple convolutional blocks
    followed by dense layers with dropout and batch normalization.
    
    Architecture:
    - 5 Convolutional Blocks with increasing filters
    - Each block: Conv2D -> BatchNorm -> Activation -> Conv2D -> BatchNorm -> Activation -> MaxPooling -> Dropout
    - Global Average Pooling
    - 3 Dense layers with BatchNorm and Dropout
    - Output layer with Softmax
    """
    
    inputs = layers.Input(shape=input_shape, name='input_layer')
    
    # ========== CONVOLUTIONAL BLOCK 1 ==========
    x = layers.Conv2D(64, (3, 3), padding='same', 
                      kernel_regularizer=regularizers.l2(1e-4),
                      name='conv1_1')(inputs)
    x = layers.BatchNormalization(name='bn1_1')(x)
    x = layers.Activation('relu', name='relu1_1')(x)
    
    x = layers.Conv2D(64, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4),
                      name='conv1_2')(x)
    x = layers.BatchNormalization(name='bn1_2')(x)
    x = layers.Activation('relu', name='relu1_2')(x)
    
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    x = layers.Dropout(0.1, name='dropout1')(x)
    
    # ========== CONVOLUTIONAL BLOCK 2 ==========
    x = layers.Conv2D(128, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4),
                      name='conv2_1')(x)
    x = layers.BatchNormalization(name='bn2_1')(x)
    x = layers.Activation('relu', name='relu2_1')(x)
    
    x = layers.Conv2D(128, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4),
                      name='conv2_2')(x)
    x = layers.BatchNormalization(name='bn2_2')(x)
    x = layers.Activation('relu', name='relu2_2')(x)
    
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    x = layers.Dropout(0.2, name='dropout2')(x)
    
    # ========== CONVOLUTIONAL BLOCK 3 ==========
    x = layers.Conv2D(256, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4),
                      name='conv3_1')(x)
    x = layers.BatchNormalization(name='bn3_1')(x)
    x = layers.Activation('relu', name='relu3_1')(x)
    
    x = layers.Conv2D(256, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4),
                      name='conv3_2')(x)
    x = layers.BatchNormalization(name='bn3_2')(x)
    x = layers.Activation('relu', name='relu3_2')(x)
    
    x = layers.Conv2D(256, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4),
                      name='conv3_3')(x)
    x = layers.BatchNormalization(name='bn3_3')(x)
    x = layers.Activation('relu', name='relu3_3')(x)
    
    x = layers.MaxPooling2D((2, 2), name='pool3')(x)
    x = layers.Dropout(0.3, name='dropout3')(x)
    
    # ========== CONVOLUTIONAL BLOCK 4 ==========
    x = layers.Conv2D(512, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4),
                      name='conv4_1')(x)
    x = layers.BatchNormalization(name='bn4_1')(x)
    x = layers.Activation('relu', name='relu4_1')(x)
    
    x = layers.Conv2D(512, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4),
                      name='conv4_2')(x)
    x = layers.BatchNormalization(name='bn4_2')(x)
    x = layers.Activation('relu', name='relu4_2')(x)
    
    x = layers.Conv2D(512, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4),
                      name='conv4_3')(x)
    x = layers.BatchNormalization(name='bn4_3')(x)
    x = layers.Activation('relu', name='relu4_3')(x)
    
    x = layers.MaxPooling2D((2, 2), name='pool4')(x)
    x = layers.Dropout(0.4, name='dropout4')(x)
    
    # ========== CONVOLUTIONAL BLOCK 5 ==========
    x = layers.Conv2D(512, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4),
                      name='conv5_1')(x)
    x = layers.BatchNormalization(name='bn5_1')(x)
    x = layers.Activation('relu', name='relu5_1')(x)
    
    x = layers.Conv2D(512, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4),
                      name='conv5_2')(x)
    x = layers.BatchNormalization(name='bn5_2')(x)
    x = layers.Activation('relu', name='relu5_2')(x)
    
    x = layers.Conv2D(512, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4),
                      name='conv5_3')(x)
    x = layers.BatchNormalization(name='bn5_3')(x)
    x = layers.Activation('relu', name='relu5_3')(x)
    
    x = layers.MaxPooling2D((2, 2), name='pool5')(x)
    x = layers.Dropout(0.5, name='dropout5')(x)
    
    # ========== GLOBAL POOLING ==========
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # ========== DENSE LAYERS (DNN) ==========
    x = layers.Dense(1024, kernel_regularizer=regularizers.l2(1e-4), name='fc1')(x)
    x = layers.BatchNormalization(name='bn_fc1')(x)
    x = layers.Activation('relu', name='relu_fc1')(x)
    x = layers.Dropout(0.5, name='dropout_fc1')(x)
    
    x = layers.Dense(512, kernel_regularizer=regularizers.l2(1e-4), name='fc2')(x)
    x = layers.BatchNormalization(name='bn_fc2')(x)
    x = layers.Activation('relu', name='relu_fc2')(x)
    x = layers.Dropout(0.4, name='dropout_fc2')(x)
    
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4), name='fc3')(x)
    x = layers.BatchNormalization(name='bn_fc3')(x)
    x = layers.Activation('relu', name='relu_fc3')(x)
    x = layers.Dropout(0.3, name='dropout_fc3')(x)
    
    # ========== OUTPUT LAYER ==========
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='CustomCNN_Cassava')
    
    return model

# ============================================================================
# BUILD AND COMPILE MODEL
# ============================================================================
print("\n" + "="*70)
print("BUILDING CUSTOM CNN-DNN MODEL")
print("="*70 + "\n")

model = build_custom_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=len(label_map))

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
)

# Display model architecture
model.summary()

# Count parameters
total_params = model.count_params()
trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ============================================================================
# LEARNING RATE SCHEDULE
# ============================================================================
def lr_schedule(epoch, lr):
    """Learning rate schedule with step decay"""
    if epoch < 15:
        return INITIAL_LEARNING_RATE
    elif epoch < 30:
        return INITIAL_LEARNING_RATE * 0.1
    elif epoch < 40:
        return INITIAL_LEARNING_RATE * 0.01
    else:
        return INITIAL_LEARNING_RATE * 0.001

# ============================================================================
# CALLBACKS
# ============================================================================
os.makedirs('models', exist_ok=True)

checkpoint = ModelCheckpoint(
    'models/custom_cnn_best_model.keras',
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

lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

callbacks = [checkpoint, reduce_lr, lr_scheduler, early_stopping]


print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70 + "\n")

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)
