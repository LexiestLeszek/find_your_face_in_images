# requirements.txt
# flask==3.0.2
# deepface==0.0.89
# opencv-python==4.9.0.80
# python-dotenv==1.0.1

from flask import Flask, request, render_template_string, send_from_directory
from deepface import DeepFace
import os
import cv2
import uuid
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# хуярим хтмл сразу тут, чтобы не было js
HTML = '''
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Поиск лиц</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&family=Montserrat:wght@500;700&display=swap" rel="stylesheet">
    <style>
        body { 
            font-family: 'Roboto', sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 800px;
            padding-top: 2rem;
        }
        .drop-zone {
            border: 3px dashed #007bff;
            border-radius: 15px;
            padding: 2rem;
            text-align: center;
            transition: all 0.3s;
            background: rgba(255,255,255,0.9);
        }
        .drop-zone:hover {
            background: rgba(0,123,255,0.1);
            transform: scale(1.02);
        }
        .carousel-item img {
            height: 400px;
            object-fit: contain;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        h1 {
            font-family: 'Montserrat', sans-serif;
            color: #2c3e50;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .emoji {
            font-size: 4rem;
        }
        .htmx-indicator {
            display: none;
        }
        form.htmx-request .htmx-indicator {
            display: inline-block;
        }
        form.htmx-request .btn-text {
            display: none;
        }
        .carousel-caption {
            font-size: 14px; /* Adjust the font size */
            padding: 8px 12px; /* Adjust padding for better spacing */
            background-color: rgba(0, 0, 0, 0.5); /* Semi-transparent background */
            border-radius: 5px; /* Rounded corners */
            bottom: 20px; /* Position from the bottom */
            left: 50%; /* Center horizontally */
            transform: translateX(-50%); /* Center horizontally */
            width: auto; /* Auto width based on content */
            max-width: 80%; /* Prevent it from being too wide */
        }

        .carousel-caption p {
            margin: 0; /* Remove default margin */
            font-weight: 500; /* Medium font weight */
        }
        .popup-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            z-index: 1000;
            cursor: pointer;
        }

        .popup-image {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            max-width: 90%;
            max-height: 90%;
            object-fit: contain;
            pointer-events: none;
        }

        .popup-image:hover {
            pointer-events: auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-center mb-4">🔍 Лицо к осмотру быстро</h1>
        
        <form hx-post="/process" hx-encoding="multipart/form-data" hx-target="#results">
            <div class="drop-zone mb-4">
                <h5 class="mb-3">📸 Загрузите эталонное фото с вашим лицом</h5>
                <input type="file" name="reference" class="form-control" accept="image/*" required>
            </div>

            <div class="drop-zone mb-4">
                <h5 class="mb-3">📁 Выберите фотографии для поиска</h5>
                <input type="file" name="targets" class="form-control" webkitdirectory directory multiple accept="image/*" required>
            </div>

            <div class="text-center">
                <button type="submit" class="btn btn-primary btn-lg px-5">
                    <span class="spinner-border spinner-border-sm htmx-indicator" role="status" aria-hidden="true"></span>
                    <span class="btn-text">🚀 Начать поиск</span>
                </button>
            </div>
        </form>

        <div id="results" class="mt-5"></div>
    </div>

    <script src="https://unpkg.com/htmx.org@1.9.10"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        function openImagePopup(src) {
            const popup = document.getElementById('imagePopup');
            const popupImg = popup.querySelector('.popup-image');
            popupImg.src = src;
            popup.style.display = 'block';
            document.body.style.overflow = 'hidden';
        }

        function closeImagePopup() {
            const popup = document.getElementById('imagePopup');
            popup.style.display = 'none';
            document.body.style.overflow = 'auto';
        }
    </script>
    <div id="imagePopup" class="popup-overlay" onclick="closeImagePopup()">
    <img class="popup-image" onclick="event.stopPropagation()">
</div>
</body>
</html>
'''

def process_images(reference_path, target_paths):
    model_name = "Facenet"
    detector_backend = "retinaface"
    threshold = 0.4

    try:
        ref_embedding = DeepFace.represent(
            img_path=reference_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=True
        )[0]["embedding"]
    except:
        return []

    matches = []
    for target in target_paths:
        try:
            result = DeepFace.verify(
                img1_path=reference_path,
                img2_path=target,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=True
            )
            if result["verified"] and result["distance"] < threshold:
                matches.append(target)
        except:
            continue

    return matches

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/process', methods=['POST'])
def handle_upload():
    # Сохраняем файлы
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Сохраняем эталонное фото
    ref_file = request.files['reference']
    ref_ext = ref_file.filename.split('.')[-1]
    ref_filename = f"ref_{timestamp}_{uuid.uuid4().hex}.{ref_ext}"
    ref_path = os.path.join(app.config['UPLOAD_FOLDER'], ref_filename)
    ref_file.save(ref_path)
    
    # Сохраняем целевые фото
    target_paths = []
    for target in request.files.getlist('targets'):
        ext = target.filename.split('.')[-1]
        filename = f"target_{timestamp}_{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        target.save(filepath)
        target_paths.append(filepath)
    
    # Обрабатываем изображения
    matches = process_images(ref_path, target_paths)
    
    # Генерируем HTML ответ
    if not matches:
        return '''
        <div class="text-center py-5">
            <div class="emoji">😢</div>
            <h3 class="mt-3">Ничего не найдено</h3>
            <p class="text-muted">Попробуйте использовать другое эталонное фото или больше целевых изображений</p>
        </div>
        '''
    else:
        carousel_items = ""
        for i, path in enumerate(matches):
            filename = os.path.basename(path)
            carousel_items += f'''
            <div class="carousel-item {'active' if i == 0 else ''}">
                <img src="/uploads/{filename}" 
                    class="d-block w-100" 
                    alt="Найдено" 
                    onclick="openImagePopup(this.src)"
                    style="cursor: pointer">
                    <div class="carousel-caption d-none d-md-block bg-dark bg-opacity-50 rounded">
                    <p>#{i+1}</p>
                </div>
            </div>
            '''
        
        return f'''
        <div class="text-center">
            <h3 class="mb-4">🎉 Найдено совпадений: {len(matches)}</h3>
            <div id="resultCarousel" class="carousel slide" data-bs-ride="carousel">
                <div class="carousel-inner">
                    {carousel_items}
                </div>
                <button class="carousel-control-prev" type="button" data-bs-target="#resultCarousel" data-bs-slide="prev">
                    <span class="carousel-control-prev-icon" ></span>
                    <span class="visually-hidden">Previous</span>
                </button>
                <button class="carousel-control-next" type="button" data-bs-target="#resultCarousel" data-bs-slide="next">
                    <span class="carousel-control-next-icon" ></span>
                    <span class="visually-hidden">Next</span>
                </button>
            </div>
        </div>
        '''

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
