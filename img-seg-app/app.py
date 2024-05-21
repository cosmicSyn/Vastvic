from flask import Flask, render_template, request, redirect, jsonify, url_for, session, abort
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

from torchvision.io.image import read_image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_segmentation_masks
from torchvision.io import read_image
from io import BytesIO
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms

weights = FCN_ResNet50_Weights.DEFAULT
model = fcn_resnet50(weights=weights)
# model.eval()

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    image_data = request.files["image"].read()
    image = Image.open(BytesIO(image_data)).convert("RGB")
    preprocess = weights.transforms()
    batch = preprocess(image).unsqueeze(0)
    # print(batch.shape)
    # print(batch.dtype)
    prediction = model(batch)["out"]
    normalized_masks = prediction.softmax(dim=1)
    num_classes = normalized_masks.shape[1]
    img1_masks = normalized_masks[0]
    class_dim = 0
    img1_all_classes_masks = img1_masks.argmax(class_dim) == torch.arange(num_classes)[:, None, None]
    masks=img1_all_classes_masks[1:]
    # print(f"img1_masks shape = {img1_masks.shape}, dtype = {img1_masks.dtype}")
    # print(f"img1_all_classes_masks = {img1_all_classes_masks.shape}, dtype = {img1_all_classes_masks.dtype}")
    img_with_all_masks = draw_segmentation_masks(batch[0], masks=img1_all_classes_masks, alpha=0.5)
    # to_pil_image(img_with_all_masks).show()
    # print(img_with_all_masks.shape)
    img_with_all_masks = img_with_all_masks.mul(255).byte().cpu()
    img_with_all_masks_pil = transforms.ToPILImage()(img_with_all_masks)
    # Define the path where the image will be saved
    output_path = "static/output_image.png"
    img_with_all_masks_pil.save(output_path)
    return render_template('index.html',display=True)
    
if __name__ == "__main__":
    app.run(debug=True)

    