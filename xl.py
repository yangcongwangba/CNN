import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
import pathlib
import numpy as np
import matplotlib.pyplot as plt

# 配置GPU
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    gpu0 = gpus[0]  # 如果有多个GPU，仅使用第0个GPU
    tf.config.experimental.set_memory_growth(gpu0, True)  # 设置GPU显存按需分配
    tf.config.set_visible_devices([gpu0], "GPU")

# 自动设置数据目录路径
data_dir = input("请输入数据集文件夹路径：")  # 输入文件夹路径，确保文件夹内按类分类
data_dir = pathlib.Path(data_dir)

# 检查数据集目录是否存在
if not data_dir.exists():
    print(f"数据集文件夹不存在: {data_dir}")
    exit(1)

# 询问是否使用预训练模型
use_pretrained = input("是否使用预训练模型？（输入 'y' 使用，'n' 不使用）：").lower()

# 数据加载和预处理
batch_size = 32
img_height = 180
img_width = 180

# 自动划分数据集
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# 获取类别名称
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"数据集类别: {class_names}")

# 数据预处理：缓存、打乱、预取数据
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 数据增强
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

# 根据是否使用预训练模型加载不同的架构
if use_pretrained == 'y':
    # 输入预训练模型的路径
    pretrained_model_path = input("请输入预训练模型的路径（例如：'pretrained_model.h5'）：")

    # 检查路径是否存在且是文件
    if not os.path.exists(pretrained_model_path):
        print(f"文件不存在: {pretrained_model_path}")
        exit(1)
    elif os.path.isdir(pretrained_model_path):
        print(f"指定路径是目录，而不是文件: {pretrained_model_path}")
        exit(1)
    else:
        print(f"使用的预训练模型路径: {pretrained_model_path}")

        # 创建基础模型并加载预训练权重
        base_model = tf.keras.applications.ResNet101(
            weights=None,  # 不加载ImageNet权重
            include_top=False,  # 不加载顶部的全连接层
            input_shape=(img_height, img_width, 3)
        )
        base_model.load_weights(pretrained_model_path)  # 加载用户指定的预训练权重
        base_model.trainable = False  # 冻结预训练模型的层

        # 构建模型
        model = models.Sequential([
            layers.InputLayer(input_shape=(img_height, img_width, 3)),  # 明确指定输入形状
            data_augmentation,  # 数据增强
            layers.Rescaling(1./255),  # Rescaling 层会自动推导输入形状
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
else:
    print("未使用预训练模型，使用默认模型。")
    # 使用较简单的CNN架构
    model = models.Sequential([
        layers.InputLayer(input_shape=(img_height, img_width, 3)),  # 明确指定输入形状
        data_augmentation,  # 数据增强
        layers.Rescaling(1./255),  # Rescaling 层会自动推导输入形状
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

model.summary()

# 编译模型
while True:
    learning_rate = input("请输入学习率（默认0.001，按回车使用默认值）：")
    if learning_rate == '':
        learning_rate = 0.001
        break
    try:
        learning_rate = float(learning_rate)
        break
    except ValueError:
        print("无效输入，请输入一个有效的数字。")

opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

# 训练模型
while True:
    epochs = input("请输入训练轮数（默认10轮，按回车使用默认值）：")
    if epochs == '':
        epochs = 10
        break
    try:
        epochs = int(epochs)
        break
    except ValueError:
        print("无效输入，请输入一个有效的数字。")

history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# 询问用户是否保存模型
while True:
    user_input = input("是否保存模型？（输入 's' 保存，'n' 不保存，'c' 继续训练）：").lower()
    if user_input == "s":
        model_save_path = input("请输入保存模型的路径（例如：'model.keras'）：")
        while not model_save_path.lower().endswith(('.h5', '.keras')):
            print("错误：模型文件格式不支持！请使用 '.h5' 或 '.keras' 格式。")
            model_save_path = input("请输入保存模型的路径（例如：'model.keras'）：")
        model.save(model_save_path)
        print(f"模型已保存到: {model_save_path}")

        # 保存类别名称
        class_names_path = input("请输入保存类别名称的路径（例如：'class_names.json'）：")
        while not class_names_path.lower().endswith('.json'):
            print("错误：类别名称文件格式不支持！请使用 '.json' 格式。")
            class_names_path = input("请输入保存类别名称的路径（例如：'class_names.json'）：")

        with open(class_names_path, 'w', encoding='utf-8') as f:
            json.dump(class_names, f)
        print(f"类别名称已保存到: {class_names_path}")
        break

    elif user_input == "n":
        print("模型未保存，类别名称未保存。")
        break
    elif user_input == "c":
        print("继续训练...")
        continue  # 继续训练
    else:
        print("无效输入，请重新输入。")

# 绘制训练过程中的准确率和损失变化
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

print("训练完成。")
