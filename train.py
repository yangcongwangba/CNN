import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models, applications
import pathlib
from tkinter import Tk, Button, Label, Entry, filedialog, messagebox, Checkbutton, BooleanVar, Scale, Frame, ttk, Radiobutton, Toplevel
import threading
from threading import Event
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 配置 GPU
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    gpu0 = gpus[0]
    tf.config.experimental.set_memory_growth(gpu0, True)
    tf.config.set_visible_devices([gpu0], "GPU")

# 创建主窗口
root = Tk()
root.title("图像分类模型训练")
root.geometry("700x830")

# 全局变量
data_dir = None
train_dir = None
val_dir = None
continue_training = False
model_to_continue = None
stop_training_event = Event()
use_pretrained_model = BooleanVar(value=False)  # 是否使用预训练模型
dataset_split_method = BooleanVar(value=True)  # True: 自动划分, False: 手动上传

# 默认值
DEFAULT_BATCH_SIZE = 32
DEFAULT_IMG_HEIGHT = 180
DEFAULT_IMG_WIDTH = 180

# 支持的预训练模型及其介绍
PRETRAINED_MODELS = {
    "MobileNetV2": {
        "class": applications.MobileNetV2,
        "description": "轻量级模型，适合移动设备和嵌入式设备。"
    },
    "ResNet50": {
        "class": applications.ResNet50,
        "description": "经典的深度残差网络，适合图像分类任务。"
    },
    "InceptionV3": {
        "class": applications.InceptionV3,
        "description": "使用 Inception 模块的深度网络，适合大规模图像分类。"
    },
    "VGG16": {
        "class": applications.VGG16,
        "description": "经典的卷积神经网络，结构简单但参数量较大。"
    },
    "EfficientNetB0": {
        "class": applications.EfficientNetB0,
        "description": "高效的卷积神经网络，性能优越。"
    },
    "DenseNet121": {
        "class": applications.DenseNet121,
        "description": "密集连接的网络，适合小数据集。"
    },
    "NASNetMobile": {
        "class": applications.NASNetMobile,
        "description": "基于神经架构搜索的网络，性能优越。"
    },
    "Xception": {
        "class": applications.Xception,
        "description": "基于深度可分离卷积的网络，适合图像分类。"
    }
}

# 选择数据集文件夹
def select_data_dir():
    global data_dir
    data_dir = filedialog.askdirectory(title="选择数据集文件夹")
    if data_dir:
        data_dir = pathlib.Path(data_dir)
        data_dir_label.config(text=f"数据集路径: {data_dir}")
    else:
        messagebox.showwarning("警告", "未选择数据集文件夹！")

# 选择训练集文件夹
def select_train_dir():
    global train_dir
    train_dir = filedialog.askdirectory(title="选择训练集文件夹")
    if train_dir:
        train_dir = pathlib.Path(train_dir)
        train_dir_label.config(text=f"训练集路径: {train_dir}")
    else:
        messagebox.showwarning("警告", "未选择训练集文件夹！")

# 选择验证集文件夹
def select_val_dir():
    global val_dir
    val_dir = filedialog.askdirectory(title="选择验证集文件夹")
    if val_dir:
        val_dir = pathlib.Path(val_dir)
        val_dir_label.config(text=f"验证集路径: {val_dir}")
    else:
        messagebox.showwarning("警告", "未选择验证集文件夹！")

# 选择继续训练的模型文件
def select_model_to_continue():
    global model_to_continue
    model_to_continue = filedialog.askopenfilename(
        title="选择继续训练的模型文件",
        filetypes=[("H5 文件", "*.h5"), ("Keras 文件", "*.keras")]
    )
    if model_to_continue:
        continue_model_label.config(text=f"继续训练模型路径: {model_to_continue}")
    else:
        messagebox.showwarning("警告", "未选择继续训练的模型文件！")

# 停止训练
def stop_training_callback():
    stop_training_event.set()
    stop_button.config(state="disabled")
    start_button.config(state="normal")
    messagebox.showinfo("提示", "训练将在当前轮次完成后停止...")

# 更新训练状态
def update_training_status(epoch, logs):
    current_epoch_label.config(text=f"当前轮数: {epoch + 1}")
    train_loss_label.config(text=f"训练损失: {logs['loss']:.4f}")
    train_acc_label.config(text=f"训练准确率: {logs['accuracy']:.4f}")
    val_loss_label.config(text=f"验证损失: {logs['val_loss']:.4f}")
    val_acc_label.config(text=f"验证准确率: {logs['val_accuracy']:.4f}")
    root.update_idletasks()

# 自定义回调以支持停止训练
class StopTrainingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if stop_training_event.is_set():
            print("训练停止")
            self.model.stop_training = True

# 创建模型（支持预训练模型）
def create_model(img_height, img_width, num_classes, use_pretrained=False, model_name="MobileNetV2"):
    if use_pretrained:
        # 获取用户选择的预训练模型
        model_info = PRETRAINED_MODELS.get(model_name, PRETRAINED_MODELS["MobileNetV2"])
        base_model = model_info["class"](
            input_shape=(img_height, img_width, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
    else:
        model = models.Sequential([
            layers.InputLayer(input_shape=(img_height, img_width, 3)),
            layers.Rescaling(1./255),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.5),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.5),
            layers.Flatten(),
            layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
    return model

# 数据增强
def create_data_augmentation():
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ])
    return data_augmentation

# 绘制训练历史图像
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 绘制训练和验证的损失曲线
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # 绘制训练和验证的准确率曲线
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    return fig

# 在弹窗中显示训练历史图像
def show_training_history(history):
    popup = Toplevel()
    popup.title("训练历史")
    popup.geometry("800x500")

    fig = plot_training_history(history)
    canvas = FigureCanvasTkAgg(fig, master=popup)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    plt.close(fig)

# 训练任务
def train_task():
    global data_dir, train_dir, val_dir, continue_training, model_to_continue
    if dataset_split_method.get():  # 自动划分数据集
        if not data_dir or not data_dir.exists():
            messagebox.showerror("错误", "数据集文件夹不存在或未选择！")
            return
    else:  # 手动上传数据集
        if not train_dir or not train_dir.exists() or not val_dir or not val_dir.exists():
            messagebox.showerror("错误", "训练集或验证集文件夹不存在或未选择！")
            return

    continue_training = continue_training_var.get()
    if continue_training and (not model_to_continue or not os.path.exists(model_to_continue)):
        messagebox.showerror("错误", "继续训练的模型文件不存在或未选择！")
        return

    # 获取用户选择的预训练模型名称
    model_name = model_combobox.get()

    # 默认设置
    batch_size = int(batch_size_entry.get() or DEFAULT_BATCH_SIZE)
    img_height = int(img_height_entry.get() or DEFAULT_IMG_HEIGHT)
    img_width = int(img_width_entry.get() or DEFAULT_IMG_WIDTH)
    validation_split = float(validation_split_scale.get() / 100.0)
    learning_rate = float(learning_rate_entry.get() or 0.001)
    epochs = int(epochs_entry.get() or 10)

    # 数据加载和预处理
    if dataset_split_method.get():  # 自动划分数据集
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=validation_split,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            label_mode='categorical'
        )

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=validation_split,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            label_mode='categorical'
        )
    else:  # 手动上传数据集
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            train_dir,
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            label_mode='categorical'
        )

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            val_dir,
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            label_mode='categorical'
        )

    # 获取类别名称
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"数据集类别: {class_names}")

    # 数据增强
    data_augmentation = create_data_augmentation()
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

    # 数据预处理：缓存、打乱、预取数据
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 创建模型
    if continue_training:
        model = tf.keras.models.load_model(model_to_continue)
    else:
        model = create_model(img_height, img_width, num_classes, use_pretrained_model.get(), model_name)

    model.summary()

    # 编译模型
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 训练模型
    stop_button.config(state="normal")
    start_button.config(state="disabled")

    callbacks = [
        tf.keras.callbacks.LambdaCallback(on_epoch_end=update_training_status),
        StopTrainingCallback(),
    ]
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks
    )

    # 询问是否保存模型
    save_model = messagebox.askyesno("保存模型", "是否保存模型？")
    if save_model:
        folder_name = filedialog.asksaveasfilename(title="保存模型文件夹", defaultextension="", filetypes=[("文件夹", "*")])
        if folder_name:
            os.makedirs(folder_name, exist_ok=True)
            model_save_path = os.path.join(folder_name, "model.keras")
            model.save(model_save_path)
            with open(os.path.join(folder_name, "class_names.json"), 'w', encoding='utf-8') as f:
                json.dump(class_names, f)
            messagebox.showinfo("成功", f"模型和类别名称已保存到: {folder_name}")

    # 显示训练历史图像
    show_training_history(history)

    # 训练结束后启用所有按钮
    data_dir_button.config(state="normal")
    train_dir_button.config(state="normal")
    val_dir_button.config(state="normal")
    continue_model_button.config(state="normal")
    batch_size_entry.config(state="normal")
    img_height_entry.config(state="normal")
    img_width_entry.config(state="normal")
    validation_split_scale.config(state="normal")
    learning_rate_entry.config(state="normal")
    epochs_entry.config(state="normal")
    model_combobox.config(state="normal")
    use_pretrained_model_checkbutton.config(state="normal")
    dataset_split_method_auto.config(state="normal")
    dataset_split_method_manual.config(state="normal")
    start_button.config(state="normal")
    stop_button.config(state="disabled")
    messagebox.showinfo("完成", "训练完成！")

# 开始训练（启动多线程）
def start_training():
    stop_training_event.clear()
    # 禁用所有按钮
    data_dir_button.config(state="disabled")
    train_dir_button.config(state="disabled")
    val_dir_button.config(state="disabled")
    continue_model_button.config(state="disabled")
    batch_size_entry.config(state="disabled")
    img_height_entry.config(state="disabled")
    img_width_entry.config(state="disabled")
    validation_split_scale.config(state="disabled")
    learning_rate_entry.config(state="disabled")
    epochs_entry.config(state="disabled")
    model_combobox.config(state="disabled")
    use_pretrained_model_checkbutton.config(state="disabled")
    dataset_split_method_auto.config(state="disabled")
    dataset_split_method_manual.config(state="disabled")
    start_button.config(state="disabled")
    stop_button.config(state="normal")
    # 启动训练线程
    training_thread = threading.Thread(target=train_task)
    training_thread.start()

# 更新模型介绍
def update_model_description(event):
    selected_model = model_combobox.get()
    model_description = PRETRAINED_MODELS.get(selected_model, {}).get("description", "无描述")
    model_description_label.config(text=f"模型介绍: {model_description}")

# 动态禁用或启用文件夹选择按钮
def toggle_dataset_selection():
    if dataset_split_method.get():  # 自动划分
        data_dir_button.config(state="normal")
        train_dir_button.config(state="disabled")
        val_dir_button.config(state="disabled")
        validation_split_scale.config(state="normal")
    else:  # 手动上传
        data_dir_button.config(state="disabled")
        train_dir_button.config(state="normal")
        val_dir_button.config(state="normal")
        validation_split_scale.config(state="disabled")

# GUI 布局
frame = Frame(root)
frame.pack(padx=10, pady=10)

# 数据集划分方式选择
Label(frame, text="数据集划分方式:").grid(row=0, column=0, padx=10, pady=10)
dataset_split_method_auto = Radiobutton(frame, text="自动划分训练集和验证集", variable=dataset_split_method, value=True, command=toggle_dataset_selection)
dataset_split_method_auto.grid(row=0, column=1, padx=10, pady=10)
dataset_split_method_manual = Radiobutton(frame, text="手动上传训练集和验证集", variable=dataset_split_method, value=False, command=toggle_dataset_selection)
dataset_split_method_manual.grid(row=0, column=2, padx=10, pady=10)

# 自动划分数据集
Label(frame, text="数据集文件夹:").grid(row=1, column=0, padx=10, pady=10)
data_dir_label = Label(frame, text="未选择", fg="red")
data_dir_label.grid(row=1, column=1, padx=10, pady=10)
data_dir_button = Button(frame, text="选择文件夹", command=select_data_dir)
data_dir_button.grid(row=1, column=2, padx=10, pady=10)

# 手动上传数据集
Label(frame, text="训练集文件夹:").grid(row=2, column=0, padx=10, pady=10)
train_dir_label = Label(frame, text="未选择", fg="red")
train_dir_label.grid(row=2, column=1, padx=10, pady=10)
train_dir_button = Button(frame, text="选择文件夹", command=select_train_dir, state="disabled")
train_dir_button.grid(row=2, column=2, padx=10, pady=10)

Label(frame, text="验证集文件夹:").grid(row=3, column=0, padx=10, pady=10)
val_dir_label = Label(frame, text="未选择", fg="red")
val_dir_label.grid(row=3, column=1, padx=10, pady=10)
val_dir_button = Button(frame, text="选择文件夹", command=select_val_dir, state="disabled")
val_dir_button.grid(row=3, column=2, padx=10, pady=10)

continue_training_var = BooleanVar()
Checkbutton(frame, text="继续训练已有模型", variable=continue_training_var).grid(row=4, column=0, padx=10, pady=10)
continue_model_label = Label(frame, text="未选择", fg="red")
continue_model_label.grid(row=4, column=1, padx=10, pady=10)
continue_model_button = Button(frame, text="选择模型文件", command=select_model_to_continue)
continue_model_button.grid(row=4, column=2, padx=10, pady=10)

use_pretrained_model_checkbutton = Checkbutton(frame, text="使用预训练模型", variable=use_pretrained_model)
use_pretrained_model_checkbutton.grid(row=5, column=0, padx=10, pady=10)

Label(frame, text="预训练模型:").grid(row=6, column=0, padx=10, pady=10)
model_combobox = ttk.Combobox(frame, values=list(PRETRAINED_MODELS.keys()))
model_combobox.grid(row=6, column=1, padx=10, pady=10)
model_combobox.set("MobileNetV2")
model_combobox.bind("<<ComboboxSelected>>", update_model_description)

model_description_label = Label(frame, text="模型介绍: 轻量级模型，适合移动设备和嵌入式设备。", wraplength=400)
model_description_label.grid(row=7, column=0, columnspan=3, padx=10, pady=10)

Label(frame, text="批量大小:").grid(row=8, column=0, padx=10, pady=10)
batch_size_entry = Entry(frame)
batch_size_entry.insert(0, str(DEFAULT_BATCH_SIZE))
batch_size_entry.grid(row=8, column=1, padx=10, pady=10)

Label(frame, text="图像高度:").grid(row=9, column=0, padx=10, pady=10)
img_height_entry = Entry(frame)
img_height_entry.insert(0, str(DEFAULT_IMG_HEIGHT))
img_height_entry.grid(row=9, column=1, padx=10, pady=10)

Label(frame, text="图像宽度:").grid(row=10, column=0, padx=10, pady=10)
img_width_entry = Entry(frame)
img_width_entry.insert(0, str(DEFAULT_IMG_WIDTH))
img_width_entry.grid(row=10, column=1, padx=10, pady=10)

# 验证集比例滑动条
Label(frame, text="验证集比例 (%):").grid(row=11, column=0, padx=10, pady=10)
validation_split_scale = Scale(frame, from_=1, to=99, orient="horizontal")
validation_split_scale.set(20)
validation_split_scale.grid(row=11, column=1, padx=10, pady=10)

Label(frame, text="学习率:").grid(row=12, column=0, padx=10, pady=10)
learning_rate_entry = Entry(frame)
learning_rate_entry.insert(0, "0.001")
learning_rate_entry.grid(row=12, column=1, padx=10, pady=10)

Label(frame, text="训练轮数:").grid(row=13, column=0, padx=10, pady=10)
epochs_entry = Entry(frame)
epochs_entry.insert(0, "10")
epochs_entry.grid(row=13, column=1, padx=10, pady=10)

# 训练状态显示
current_epoch_label = Label(frame, text="当前轮数: 0")
current_epoch_label.grid(row=14, column=0, padx=10, pady=10)
train_loss_label = Label(frame, text="训练损失: 0.0000")
train_loss_label.grid(row=14, column=1, padx=10, pady=10)
train_acc_label = Label(frame, text="训练准确率: 0.0000")
train_acc_label.grid(row=14, column=2, padx=10, pady=10)
val_loss_label = Label(frame, text="验证损失: 0.0000")
val_loss_label.grid(row=15, column=0, padx=10, pady=10)
val_acc_label = Label(frame, text="验证准确率: 0.0000")
val_acc_label.grid(row=15, column=1, padx=10, pady=10)

# 开始训练和停止训练按钮
start_button = Button(frame, text="开始训练", command=start_training)
start_button.grid(row=16, column=1, padx=10, pady=20)
stop_button = Button(frame, text="停止训练", command=stop_training_callback, state="disabled")
stop_button.grid(row=16, column=2, padx=10, pady=20)

# 运行主循环
root.mainloop()