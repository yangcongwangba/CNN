import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
import pathlib
from tkinter import Tk, Button, Label, Entry, filedialog, messagebox, Checkbutton, BooleanVar, Scale, Frame
import threading
from threading import Lock, Event

# 配置GPU
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    gpu0 = gpus[0]  # 如果有多个GPU，仅使用第0个GPU
    tf.config.experimental.set_memory_growth(gpu0, True)  # 设置GPU显存按需分配
    tf.config.set_visible_devices([gpu0], "GPU")

# 创建主窗口
root = Tk()
root.title("图像分类模型训练")
root.geometry("600x600")

# 全局变量
data_dir = None
pretrained_model_path = None
use_pretrained = False
continue_training = False
model_to_continue = None
stop_training_event = Event()  # 用于控制是否停止训练

# 默认值
DEFAULT_BATCH_SIZE = 32
DEFAULT_IMG_HEIGHT = 180
DEFAULT_IMG_WIDTH = 180

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
    stop_training_event.set()  # 设置停止事件
    stop_button.config(state="disabled")  # 禁用停止按钮
    start_button.config(state="normal")  # 启用开始训练按钮
    messagebox.showinfo("提示", "训练将在当前轮次完成后停止...")

# 更新训练参数显示
def update_training_status(epoch, logs):
    current_epoch_label.config(text=f"当前轮数: {epoch + 1}")
    train_loss_label.config(text=f"训练损失: {logs['loss']:.4f}")
    train_acc_label.config(text=f"训练准确率: {logs['accuracy']:.4f}")
    val_loss_label.config(text=f"验证损失: {logs['val_loss']:.4f}")
    val_acc_label.config(text=f"验证准确率: {logs['val_accuracy']:.4f}")
    root.update_idletasks()  # 更新界面

# 自定义回调以支持停止训练
class StopTrainingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if stop_training_event.is_set():  # 检查停止事件
            print("训练停止")
            self.model.stop_training = True

# 训练任务
def train_task():
    global data_dir, pretrained_model_path, use_pretrained, continue_training, model_to_continue

    if not data_dir or not data_dir.exists():
        messagebox.showerror("错误", "数据集文件夹不存在或未选择！")
        return

    use_pretrained = use_pretrained_var.get()
    if use_pretrained and (not pretrained_model_path or not os.path.exists(pretrained_model_path)):
        messagebox.showerror("错误", "预训练模型文件不存在或未选择！")
        return

    continue_training = continue_training_var.get()
    if continue_training and (not model_to_continue or not os.path.exists(model_to_continue)):
        messagebox.showerror("错误", "继续训练的模型文件不存在或未选择！")
        return

    # 获取用户输入的参数，如果未输入则使用默认值
    batch_size = int(batch_size_entry.get() or DEFAULT_BATCH_SIZE)
    img_height = int(img_height_entry.get() or DEFAULT_IMG_HEIGHT)
    img_width = int(img_width_entry.get() or DEFAULT_IMG_WIDTH)
    validation_split = float(validation_split_scale.get() / 100.0)
    learning_rate = float(learning_rate_entry.get() or 0.001)
    epochs = int(epochs_entry.get() or 10)

    # 数据加载和预处理
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
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

    # 如果继续训练，加载已有模型
    if continue_training:
        model = tf.keras.models.load_model(model_to_continue)
    else:
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
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

    # 训练模型
    stop_button.config(state="normal")  # 启用停止按钮
    start_button.config(state="disabled")  # 禁用开始按钮

    callbacks = [
        tf.keras.callbacks.LambdaCallback(on_epoch_end=update_training_status),
        StopTrainingCallback()  # 使用自定义回调来停止训练
    ]
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks
    )

    # 训练结束后，询问是否保存模型
    if stop_training_event.is_set() or history.history['accuracy'][-1] >= 0.99:
        save_model = messagebox.askyesno("保存模型", "训练已停止，是否保存模型？")
        if save_model:
            folder_name = filedialog.asksaveasfilename(title="保存模型文件夹", defaultextension="", filetypes=[("文件夹", "*")])
            if folder_name:
                os.makedirs(folder_name, exist_ok=True)
                model_save_path = os.path.join(folder_name, "model.keras")
                model.save(model_save_path)
                with open(os.path.join(folder_name, "class_names.json"), 'w', encoding='utf-8') as f:
                    json.dump(class_names, f)
                messagebox.showinfo("成功", f"模型和类别名称已保存到: {folder_name}")

    stop_button.config(state="disabled")
    start_button.config(state="normal")
    messagebox.showinfo("完成", "训练完成！")

# 开始训练（启动多线程）
def start_training():
    stop_training_event.clear()  # 重置停止事件
    training_thread = threading.Thread(target=train_task)
    training_thread.start()

# GUI 布局
frame = Frame(root)
frame.pack(padx=10, pady=10)

Label(frame, text="数据集文件夹:").grid(row=0, column=0, padx=10, pady=10)
data_dir_label = Label(frame, text="未选择", fg="red")
data_dir_label.grid(row=0, column=1, padx=10, pady=10)
Button(frame, text="选择文件夹", command=select_data_dir).grid(row=0, column=2, padx=10, pady=10)

use_pretrained_var = BooleanVar()
Checkbutton(frame, text="使用预训练模型", variable=use_pretrained_var).grid(row=1, column=0, padx=10, pady=10)
pretrained_model_label = Label(frame, text="未选择", fg="red")
pretrained_model_label.grid(row=1, column=1, padx=10, pady=10)
Button(frame, text="选择模型文件", command=select_pretrained_model).grid(row=1, column=2, padx=10, pady=10)

continue_training_var = BooleanVar()
Checkbutton(frame, text="继续训练已有模型", variable=continue_training_var).grid(row=2, column=0, padx=10, pady=10)
continue_model_label = Label(frame, text="未选择", fg="red")
continue_model_label.grid(row=2, column=1, padx=10, pady=10)
Button(frame, text="选择模型文件", command=select_model_to_continue).grid(row=2, column=2, padx=10, pady=10)

Label(frame, text="批量大小:").grid(row=3, column=0, padx=10, pady=10)
batch_size_entry = Entry(frame)
batch_size_entry.insert(0, str(DEFAULT_BATCH_SIZE))
batch_size_entry.grid(row=3, column=1, padx=10, pady=10)

Label(frame, text="图像高度:").grid(row=4, column=0, padx=10, pady=10)
img_height_entry = Entry(frame)
img_height_entry.insert(0, str(DEFAULT_IMG_HEIGHT))
img_height_entry.grid(row=4, column=1, padx=10, pady=10)

Label(frame, text="图像宽度:").grid(row=5, column=0, padx=10, pady=10)
img_width_entry = Entry(frame)
img_width_entry.insert(0, str(DEFAULT_IMG_WIDTH))
img_width_entry.grid(row=5, column=1, padx=10, pady=10)

Label(frame, text="验证集比例 (%):").grid(row=6, column=0, padx=10, pady=10)
validation_split_scale = Scale(frame, from_=0, to=100, orient="horizontal")
validation_split_scale.set(20)
validation_split_scale.grid(row=6, column=1, padx=10, pady=10)

Label(frame, text="学习率:").grid(row=7, column=0, padx=10, pady=10)
learning_rate_entry = Entry(frame)
learning_rate_entry.insert(0, "0.001")
learning_rate_entry.grid(row=7, column=1, padx=10, pady=10)

Label(frame, text="训练轮数:").grid(row=8, column=0, padx=10, pady=10)
epochs_entry = Entry(frame)
epochs_entry.insert(0, "10")
epochs_entry.grid(row=8, column=1, padx=10, pady=10)

# 训练状态显示
current_epoch_label = Label(frame, text="当前轮数: 0")
current_epoch_label.grid(row=9, column=0, padx=10, pady=10)
train_loss_label = Label(frame, text="训练损失: 0.0000")
train_loss_label.grid(row=9, column=1, padx=10, pady=10)
train_acc_label = Label(frame, text="训练准确率: 0.0000")
train_acc_label.grid(row=9, column=2, padx=10, pady=10)
val_loss_label = Label(frame, text="验证损失: 0.0000")
val_loss_label.grid(row=10, column=0, padx=10, pady=10)
val_acc_label = Label(frame, text="验证准确率: 0.0000")
val_acc_label.grid(row=10, column=1, padx=10, pady=10)

# 开始训练和停止训练按钮
start_button = Button(frame, text="开始训练", command=start_training)
start_button.grid(row=11, column=1, padx=10, pady=20)
stop_button = Button(frame, text="停止训练", command=stop_training_callback, state="disabled")
stop_button.grid(row=11, column=2, padx=10, pady=20)

# 运行主循环
root.mainloop()
