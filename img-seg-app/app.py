from flask import Flask, render_template, request, redirect,  url_for, session, abort

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.io.image import read_image
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_segmentation_masks

from io import BytesIO
from PIL import Image
from io import BytesIO

from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
t = transforms.Compose([transforms.ToTensor()])
weights = FCN_ResNet50_Weights.DEFAULT
transforms = weights.transforms(resize_size=None)
model = fcn_resnet50(weights=weights, progress=False)

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    image_data = request.files["image"].read()
    dog1 = Image.open(BytesIO(image_data)).convert("RGB")
    dog1 = t(dog1)
    dog_list = [dog1]
    batch = torch.stack([transforms(d) for d in dog_list])
    output = model(batch)['out']

    normalized_masks = torch.nn.functional.softmax(output, dim=1)
    num_classes = normalized_masks.shape[1]
    dog1_masks = normalized_masks[0]
    class_dim = 1
    all_classes_masks = normalized_masks.argmax(class_dim) == torch.arange(num_classes)[:, None, None, None]
    print(f"shape = {all_classes_masks.shape}, dtype = {all_classes_masks.dtype}")
    # The first dimension is the classes now, so we need to swap it
    all_classes_masks = all_classes_masks.swapaxes(0, 1)

    dogs_with_masks = [
        draw_segmentation_masks(img, masks=mask, alpha=.6)
        for img, mask in zip(dog_list, all_classes_masks)
    ]
    
    final_img = to_pil_image(dogs_with_masks[0])
    output_path = "static/output_image.png"
    final_img.save(output_path)
    return render_template('index.html',display=True)
    
if __name__ == "__main__":
    app.run(debug=True)

    
