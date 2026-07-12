"""
Small training harness to fine-tune a MobileNetV2 backbone for deepfake detection.

Usage:
  python scripts/train.py --data_dir path/to/dataset --epochs 5 --batch_size 32

Dataset layout expected:
  dataset/
    real/
      img1.jpg
      img2.jpg
    fake/
      imgA.jpg
      imgB.jpg

This script saves the trained model to `models/deepfake_classifier.h5`.
"""
import argparse
import os
import tensorflow as tf


def build_model(input_shape=(224, 224, 3)):
    base = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=base.input, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--out', default='models/deepfake_classifier.h5')
    args = p.parse_args()

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        args.data_dir,
        labels='inferred',
        label_mode='binary',
        image_size=(224, 224),
        batch_size=args.batch_size,
        shuffle=True,
        validation_split=0.1,
        subset='training',
        seed=123,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        args.data_dir,
        labels='inferred',
        label_mode='binary',
        image_size=(224, 224),
        batch_size=args.batch_size,
        shuffle=True,
        validation_split=0.1,
        subset='validation',
        seed=123,
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.out, save_best_only=True, monitor='val_accuracy', mode='max'),
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    model.save(args.out)
    print('Saved model to', args.out)


if __name__ == '__main__':
    main()
