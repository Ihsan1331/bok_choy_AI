from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from PIL import Image
import os

# Initialize Flask app
app = Flask(__name__)

# Load your PyTorch model
model = torch.load('resnet50.pth', map_location=torch.device('cpu'))
model.eval()

# Define transformation for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image
        image = Image.open(filepath)
        image = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            class_index = predicted.item()
            # Replace with actual class labels
            class_labels = ['Healthy', 'Disease1', 'Disease2']
            result = class_labels[class_index]
        
        return render_template('result.html', result=result, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
