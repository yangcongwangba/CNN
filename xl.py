import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Button, Label, Entry, filedialog, messagebox, Checkbutton, BooleanVar

# 配置GPU
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    gpu0 = gpus[0]  # 如果有多个GPU，仅使用第0个GPU
    tf.config.experimental.set_memory_growth(gpu0, True)  # 设置GPU显存按需分配
    tf.config.set_visible_devices([gpu0], "GPU")

# 创建主窗口
root = Tk()
root.title("图像分类模型训练")
root.geometry("400x300")

# 全局变量
data_dir = None
pretrained_model_path = None
use_pretrained = False

# 选择数据集文件夹
def select_data_dir():
    global data_dir
    data_dir = filedialog.askdirectory(title="选择数据集文件夹")
    if data_dir:
        data_dir = pathlib.Path(data_dir)
        data_dir_label.config(text=f"数据集路径: {data_dir}")
    else:
        messagebox.showwarning("警告", "未选择数据集文件夹！")

# 选择预训练模型文件
def select_pretrained_model():
    global pretrained_model_path
    pretrained_model_path = filedialog.askopenfilename(
        title="选择预训练模型文件",
        filetypes=[("H5 文件", "*.h5"), ("Keras 文件", "*.keras")]
    )
    if pretrained_model_path:
        pretrained_model_label.config(text=f"预训练模型路径: {pretrained_model_path}")
    else:
        messagebox.showwarning("警告", "未选择预训练模型文件！")

# 开始训练
def start_training():
    global data_dir, pretrained_model_path, use_pretrained

    if not data_dir or not data_dir.exists():
        messagebox.showerror("错误", "数据集文件夹不存在或未选择！")
        return

    use_pretrained = use_pretrained_var.get()
    if use_pretrained and (not pretrained_model_path or not os.path.exists(pretrained_model_path)):
        messagebox.showerror("错误", "预训练模型文件不存在或未选择！")
        return

    # 数据加载和预处理
    batch_size = 32
    img_height = 180
    img_width = 180

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
    if use_pretrained:
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
    learning_rate = float(learning_rate_entry.get() or 0.001)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

    # 训练模型
    epochs = int(epochs_entry.get() or 10)
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    # 保存模型和类别名称
    save_model = messagebox.askyesno("保存模型", "是否保存模型？")
    if save_model:
        model_save_path = filedialog.asksaveasfilename(
            title="保存模型",
            defaultextension=".keras",
            filetypes=[("H5 文件", "*.h5"), ("Keras 文件", "*.keras")]
        )
        if model_save_path:
            model.save(model_save_path)
            messagebox.showinfo("成功", f"模型已保存到: {model_save_path}")

            # 保存类别名称
            class_names_path = filedialog.asksaveasfilename(
                title="保存类别名称",
                defaultextension=".json",
                filetypes=[("JSON 文件", "*.json")]
            )
            if class_names_path:
                with open(class_names_path, 'w', encoding='utf-8') as f:
                    json.dump(class_names, f)
                messagebox.showinfo("成功", f"类别名称已保存到: {class_names_path}")

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

    messagebox.showinfo("完成", "训练完成！")

# GUI 布局
Label(root, text="数据集文件夹:").grid(row=0, column=0, padx=10, pady=10)
data_dir_label = Label(root, text="未选择", fg="red")
data_dir_label.grid(row=0, column=1, padx=10, pady=10)
Button(root, text="选择文件夹", command=select_data_dir).grid(row=0, column=2, padx=10, pady=10)

use_pretrained_var = BooleanVar()
Checkbutton(root, text="使用预训练模型", variable=use_pretrained_var).grid(row=1, column=0, padx=10, pady=10)
pretrained_model_label = Label(root, text="未选择", fg="red")
pretrained_model_label.grid(row=1, column=1, padx=10, pady=10)
Button(root, text="选择模型文件", command=select_pretrained_model).grid(row=1, column=2, padx=10, pady=10)

Label(root, text="学习率:").grid(row=2, column=0, padx=10, pady=10)
learning_rate_entry = Entry(root)
learning_rate_entry.grid(row=2, column=1, padx=10, pady=10)

Label(root, text="训练轮数:").grid(row=3, column=0, padx=10, pady=10)
epochs_entry = Entry(root)
epochs_entry.grid(row=3, column=1, padx=10, pady=10)

Button(root, text="开始训练", command=start_training).grid(row=4, column=1, padx=10, pady=20)

# 运行主循环
root.mainloop()