import os
import json
import time
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import tempfile
import base64
import cv2

# 初始化 Flask 应用
app = Flask(__name__)

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'h5', 'keras', 'json', 'jpg', 'jpeg', 'png'}

# 检查文件类型是否允许
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 清空缓存文件夹
def clear_cache():
    cache_folder = 'temp'  # 缓存文件夹路径
    if os.path.exists(cache_folder):
        for file_name in os.listdir(cache_folder):
            file_path = os.path.join(cache_folder, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"已删除缓存文件: {file_path}")
            except Exception as e:
                print(f"删除缓存文件失败: {e}")
    else:
        print("缓存文件夹不存在，无需清理")

# 智能加载模型
def load_model_smart(model_path):
    try:
        # 尝试加载完整模型
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"加载模型失败: {e}")
        print(f"模型文件路径: {model_path}")
        if os.path.exists(model_path):
            print("模型文件存在")
            # 尝试读取文件内容
            try:
                with open(model_path, 'rb') as f:
                    print(f"文件大小: {len(f.read())} 字节")
            except Exception as e:
                print(f"读取文件内容失败: {e}")
        else:
            print("模型文件不存在")
        return None

# 加载类别名称
def load_class_names(class_names_path):
    try:
        with open(class_names_path, 'r', encoding='utf-8') as f:
            class_names = json.load(f)
        return class_names
    except Exception as e:
        print(f"加载类别名称失败: {e}")
        return None

# 预测函数
def predict_image(model, image_path, class_names=None):
    try:
        # 从模型中获取输入形状
        input_shape = model.input_shape[1:3]  # 获取 (height, width)
        print(f"模型输入形状: {input_shape}")

        # 加载图片并调整大小
        img = image.load_img(image_path, target_size=input_shape)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # 进行预测
        predictions = model.predict(img_array)
        probabilities = tf.nn.softmax(predictions[0]).numpy()  # 应用 softmax
        predicted_class_id = np.argmax(probabilities)
        probability = probabilities[predicted_class_id] * 100  # 转换为百分比

        # 如果没有类别名称，使用索引值作为类别名称
        if class_names is None:
            predicted_class_name = f"Class_{predicted_class_id}"
            predicted_probabilities = {
                f"Class_{i}": float(probabilities[i] * 100) for i in range(len(probabilities))  # 转换为百分比
            }
        else:
            predicted_class_name = class_names[predicted_class_id]
            predicted_probabilities = {
                class_names[i]: float(probabilities[i] * 100) for i in range(len(class_names))  # 转换为百分比
            }

        # 在图片上绘制预测名称
        original_image = cv2.imread(image_path)
        label = f"{predicted_class_name} {probability:.2f}%"

        # 获取图片尺寸
        height, width, _ = original_image.shape

        # 设置文本位置和字体大小
        font_scale = 1  # 字体大小
        thickness = 2  # 字体粗细
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = 10  # 文本左上角 x 坐标
        text_y = height - 10  # 文本左上角 y 坐标（放在图片底部）

        # 绘制文本（无背景）
        cv2.putText(
            original_image,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 0),  # 文本颜色（绿色）
            thickness
        )

        # 将处理后的图片保存为临时文件
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            output_image_path = temp_file.name
            cv2.imwrite(output_image_path, original_image)

        # 将图片编码为 Base64 字符串
        with open(output_image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        # 删除临时文件
        os.remove(output_image_path)

        return predicted_class_name, float(probability), predicted_probabilities, img_base64
    except Exception as e:
        print(f"预测失败: {e}")
        raise

@app.route('/')
def index():
    """
    渲染 index.html 页面。
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 检查上传的文件是否存在
        files = request.files

        # 处理图片文件
        if 'image_path' not in files or not files['image_path'].filename:
            return jsonify({'error': '请上传图片文件'}), 400
        image_file = files['image_path']
        if not allowed_file(image_file.filename):
            return jsonify({'error': '图片文件格式不正确'}), 400

        # 处理模型文件
        if 'model_path' not in files or not files['model_path'].filename:
            return jsonify({'error': '请上传模型文件'}), 400
        model_file = files['model_path']
        if not allowed_file(model_file.filename):
            return jsonify({'error': '模型文件格式不正确'}), 400

        # 处理类别名称文件（可选）
        class_names = None
        if 'class_names' in files and files['class_names'].filename:
            class_names_file = files['class_names']
            if allowed_file(class_names_file.filename):
                # 保存上传的文件到临时路径
                temp_folder = 'temp'
                os.makedirs(temp_folder, exist_ok=True)
                class_names_path = os.path.join(temp_folder, secure_filename(class_names_file.filename))
                class_names_file.save(class_names_path)
                # 加载类别名称
                class_names = load_class_names(class_names_path)

        # 保存上传的文件到临时路径
        temp_folder = 'temp'
        os.makedirs(temp_folder, exist_ok=True)

        image_path = os.path.join(temp_folder, secure_filename(image_file.filename))
        model_path = os.path.join(temp_folder, secure_filename(model_file.filename))

        image_file.save(image_path)
        model_file.save(model_path)

        # 加载模型
        model = load_model_smart(model_path)
        if not model:
            return jsonify({'error': '模型加载失败'}), 500

        # 进行预测
        predicted_class_name, probability, predicted_probabilities, img_base64 = predict_image(
            model, image_path, class_names
        )

        # 返回预测结果
        result = {
            'predicted_class_name': predicted_class_name,
            'probability': round(probability, 2),  # 保留两位小数
            'predicted_probabilities': predicted_probabilities,
            'image_base64': img_base64
        }

        # 清理临时文件
        clear_cache()

        return jsonify(result)

    except Exception as e:
        # 清理临时文件
        clear_cache()
        return jsonify({'error': f'发生错误: {str(e)}'}), 500

# 关闭TensorFlow的调试信息
tf.get_logger().setLevel('ERROR')

if __name__ == "__main__":
    # 启动时清空缓存
    clear_cache()
    app.run(host='0.0.0.0', port=5000)