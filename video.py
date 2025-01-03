import cv2
import numpy as np
import tensorflow as tf
import json
import tempfile
import shutil
from tkinter import Tk, Button, Label, filedialog, messagebox, ttk
from threading import Thread, Event


class VideoRecognizer:
    def __init__(self):
        self.model = None
        self.class_names = None
        self.stop_event = Event()

    def load_model(self):
        model_path = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("Keras 文件", "*.keras"), ("H5 文件", "*.h5")]
        )
        if model_path:
            try:
                self.model = tf.keras.models.load_model(model_path)
                messagebox.showinfo("成功", "模型加载成功！")
                return model_path  # 返回模型路径
            except Exception as e:
                messagebox.showerror("错误", f"加载模型失败: {e}")
        else:
            messagebox.showwarning("警告", "未选择模型文件！")
        return None

    def load_class_names(self):
        class_names_path = filedialog.askopenfilename(
            title="选择类别名称文件",
            filetypes=[("JSON 文件", "*.json")]
        )
        if class_names_path:
            try:
                with open(class_names_path, 'r', encoding='utf-8') as f:
                    self.class_names = json.load(f)
                messagebox.showinfo("成功", "类别名称加载成功！")
                return class_names_path  # 返回类别名称路径
            except Exception as e:
                messagebox.showerror("错误", f"加载类别名称失败: {e}")
        else:
            messagebox.showwarning("警告", "未选择类别名称文件！")
        return None

    @staticmethod
    def preprocess_image(image, target_size=(180, 180)):
        image = cv2.resize(image, target_size)  # 调整尺寸
        image = image / 255.0  # 归一化
        image = np.expand_dims(image, axis=0)  # 添加批次维度
        return image

    def recognize_video(self, video_path, progress_callback=None):
        if self.model is None:
            messagebox.showerror("错误", "请先加载模型！")
            return
        if self.class_names is None:
            messagebox.showerror("错误", "请先加载类别名称！")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("错误", "无法打开视频文件！")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 创建临时文件
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_path = temp_file.name
        temp_file.close()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

        current_frame = 0
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = self.preprocess_image(frame)
            predictions = self.model.predict(processed_frame)
            predicted_class_index = np.argmax(predictions)
            predicted_class = self.class_names[predicted_class_index]
            confidence = np.max(predictions)

            label = f"{predicted_class} {confidence:.2f}"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 调整视频窗口大小
            resized_frame = cv2.resize(frame, (640, 360))  # 调整为 640x360
            cv2.imshow("Video Recognition", resized_frame)
            out.write(frame)

            current_frame += 1
            if progress_callback:
                progress_callback(current_frame, total_frames)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_event.set()
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # 识别完成后提示用户选择保存路径
        save_path = filedialog.asksaveasfilename(
            title="保存输出视频",
            defaultextension=".mp4",
            filetypes=[("视频文件", "*.mp4")]
        )
        if save_path:
            shutil.move(temp_path, save_path)
            messagebox.showinfo("完成", f"视频已保存到: {save_path}")
        else:
            messagebox.showwarning("警告", "未选择保存路径，临时文件将被删除！")

    def stop_recognition(self):
        self.stop_event.set()


class VideoRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.recognizer = VideoRecognizer()
        self.video_path = None
        self.model_path = None
        self.class_names_path = None
        self.setup_ui()

    def setup_ui(self):
        self.root.title("视频识别工具")
        self.root.geometry("500x400")

        Label(self.root, text="1. 加载模型文件").pack(pady=5)
        self.model_path_label = Label(self.root, text="未选择", fg="red")
        self.model_path_label.pack(pady=5)
        Button(self.root, text="选择模型文件", command=self.load_model).pack(pady=5)

        Label(self.root, text="2. 加载类别名称文件").pack(pady=5)
        self.class_names_path_label = Label(self.root, text="未选择", fg="red")
        self.class_names_path_label.pack(pady=5)
        Button(self.root, text="选择类别名称文件", command=self.load_class_names).pack(pady=5)

        Label(self.root, text="3. 选择视频文件").pack(pady=5)
        self.video_path_label = Label(self.root, text="未选择", fg="red")
        self.video_path_label.pack(pady=5)
        Button(self.root, text="选择视频文件", command=self.select_video_file).pack(pady=5)

        Label(self.root, text="4. 开始识别").pack(pady=10)
        Button(self.root, text="开始识别", command=self.start_recognition).pack(pady=10)
        Button(self.root, text="停止识别", command=self.stop_recognition).pack(pady=10)

        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(pady=10)

    def load_model(self):
        self.model_path = self.recognizer.load_model()
        if self.model_path:
            self.model_path_label.config(text=f"模型路径: {self.model_path}")

    def load_class_names(self):
        self.class_names_path = self.recognizer.load_class_names()
        if self.class_names_path:
            self.class_names_path_label.config(text=f"类别名称路径: {self.class_names_path}")

    def select_video_file(self):
        self.video_path = self._select_file(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov")],
            label=self.video_path_label
        )

    @staticmethod
    def _select_file(title, filetypes, label, save=False):
        if save:
            file_path = filedialog.asksaveasfilename(title=title, defaultextension=filetypes[0][1][1:], filetypes=filetypes)
        else:
            file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        if file_path:
            label.config(text=f"路径: {file_path}")
            return file_path
        else:
            messagebox.showwarning("警告", "未选择文件！")
            return None

    def start_recognition(self):
        if self.video_path is None:
            messagebox.showerror("错误", "请先选择视频文件！")
            return
        self.progress["value"] = 0
        self.recognizer.stop_event.clear()
        Thread(target=self.recognizer.recognize_video, args=(self.video_path, self.update_progress)).start()

    def stop_recognition(self):
        self.recognizer.stop_recognition()

    def update_progress(self, current_frame, total_frames):
        progress_value = (current_frame / total_frames) * 100
        self.progress["value"] = progress_value
        self.root.update_idletasks()


if __name__ == "__main__":
    root = Tk()
    app = VideoRecognitionApp(root)
    root.mainloop()