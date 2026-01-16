# import os
# from uuid import uuid4
# import numpy as np
# from PIL import Image
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import transforms as T, models
# from torchvision.models import ResNet18_Weights
# import cv2
# import requests
# import base64
# from PIL import Image
# import io

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# transform_detect = T.Compose([
#     T.Resize((224, 224)),
#     T.ToTensor(),
#     T.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])

# transform_segment = T.Compose([
#     T.Resize((224, 224)),
#     T.ToTensor(),
#     T.Normalize([0.5]*3, [0.5]*3)
# ])

# transform_classify = transform_segment 


# # Stage 1: Tumor Detection

# def get_detection_model():
#     model = models.resnet18(weights=None)
#     model.fc = nn.Linear(model.fc.in_features, 2)
#     model.load_state_dict(torch.load("core/model_files/stage1_detector_yes_no.pth", map_location=device))
#     return model.to(device).eval()

# def detect_tumor(image_path):
#     try:
#         print("🧠 Loading detection model...")
#         model = get_detection_model()

#         img = Image.open(image_path).convert("RGB")
#         tensor = transform_detect(img).unsqueeze(0).to(device)

#         with torch.no_grad():
#             output = model(tensor)
#             _, pred = torch.max(output, 1)
#             result = "yes" if pred.item() == 1 else "no"
#             print("🎯 Detection:", result)

#         torch.cuda.empty_cache()
#         return result

#     except Exception as e:
#         print(f"❌ Detection error: {e}")
#         return None


# # Stage 2: Tumor Classification

# class_labels = ['glioma', 'meningioma', 'pituitary']

# def get_classification_model():
#     model = models.resnet18(weights=None)
#     model.fc = nn.Linear(model.fc.in_features, 3)
#     model.load_state_dict(torch.load("core/model_files/stage2_tumor_classifier.pth", map_location=device))
#     return model.to(device).eval()

# def classify_tumor(image_path):
#     try:
#         print("🧠 Loading classification model...")
#         model = get_classification_model()

#         img = Image.open(image_path).convert("RGB")
#         tensor = transform_classify(img).unsqueeze(0).to(device)

#         with torch.no_grad():
#             output = model(tensor)
#             _, pred = torch.max(output, 1)
#             label = class_labels[pred.item()]
#             print("📌 Classification:", label)

#         torch.cuda.empty_cache()
#         return label

#     except Exception as e:
#         print(f"❌ Classification error: {e}")
#         return None


# # Stage 3: Tumor Segmentation

# class UNet(nn.Module):
#     def __init__(self):
#         super(UNet, self).__init__()

#         def conv_block(in_c, out_c):
#             return nn.Sequential(
#                 nn.Conv2d(in_c, out_c, 3, padding=1),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(out_c, out_c, 3, padding=1),
#                 nn.ReLU(inplace=True)
#             )

#         self.enc1 = conv_block(3, 64)
#         self.enc2 = conv_block(64, 128)
#         self.enc3 = conv_block(128, 256)
#         self.enc4 = conv_block(256, 512)
#         self.pool = nn.MaxPool2d(2)
#         self.bottleneck = conv_block(512, 1024)

#         self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
#         self.dec4 = conv_block(1024, 512)

#         self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
#         self.dec3 = conv_block(512, 256)

#         self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
#         self.dec2 = conv_block(256, 128)

#         self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
#         self.dec1 = conv_block(128, 64)

#         self.final = nn.Conv2d(64, 1, 1)

#     def forward(self, x):
#         e1 = self.enc1(x)
#         e2 = self.enc2(self.pool(e1))
#         e3 = self.enc3(self.pool(e2))
#         e4 = self.enc4(self.pool(e3))
#         b = self.bottleneck(self.pool(e4))

#         d4 = self.upconv4(b)
#         d4 = self.dec4(torch.cat([d4, e4], dim=1))

#         d3 = self.upconv3(d4)
#         d3 = self.dec3(torch.cat([d3, e3], dim=1))

#         d2 = self.upconv2(d3)
#         d2 = self.dec2(torch.cat([d2, e2], dim=1))

#         d1 = self.upconv1(d2)
#         d1 = self.dec1(torch.cat([d1, e1], dim=1))

#         return torch.sigmoid(self.final(d1))

# def get_segmentation_model():
#     model = UNet()
#     model.load_state_dict(torch.load("core/model_files/best_unet_segmentation.pth", map_location=device))
#     model.to(device).eval()

#     gradients, activations = [], []

#     def save_gradient(module, grad_input, grad_output):
#         gradients.clear()
#         gradients.append(grad_output[0])

#     def save_activation(module, input, output):
#         activations.clear()
#         activations.append(output)

#     model.enc4.register_forward_hook(save_activation)
#     model.enc4.register_full_backward_hook(save_gradient)

#     return model, gradients, activations

# def segment_tumor(image_path):
#     try:
#         model, _, _ = get_segmentation_model()

#         image = Image.open(image_path).convert("RGB")
#         original_size = image.size

#         input_tensor = transform_segment(image).unsqueeze(0).to(device)
#         with torch.no_grad():
#             output = model(input_tensor)
#             mask = output.squeeze().cpu().numpy()

#         mask_binary = (mask > 0.5).astype(np.uint8)
#         mask_img = Image.fromarray(mask_binary * 255).resize(original_size)
#         mask_filename = f"{uuid4().hex}.png"
#         save_path = os.path.join("media", "segmented", mask_filename)
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         mask_img.save(save_path)
#         tumor_area = int(np.sum(mask_binary))
#         center = find_tumor_center(mask_binary)
#         if center:
#             tumor_center_x, tumor_center_y = center
#         else:
#             tumor_center_x, tumor_center_y = -1, -1  

#         torch.cuda.empty_cache()
#         return f"segmented/{mask_filename}", tumor_area, (tumor_center_x, tumor_center_y)

#     except Exception as e:
#         print(f"❌ Segmentation error: {e}")
#         return None, None, (-1, -1)




# # GradCAM & Heatmap

# def compute_gradcam(image_tensor, model, gradients, activations):
#     model.eval()
#     image_tensor = image_tensor.unsqueeze(0).to(device)
#     image_tensor.requires_grad = True

#     output = model(image_tensor)
#     score = output[0, 0].max()
#     model.zero_grad()
#     score.backward()

#     grad = gradients[0].detach()
#     act = activations[0].detach()
#     weights = grad.mean(dim=(2, 3), keepdim=True)
#     cam = (weights * act).sum(dim=1, keepdim=True)
#     cam = F.relu(cam)

#     cam = cam.squeeze().cpu().numpy()
#     cam = (cam - cam.min()) / (cam.max() + 1e-8)
#     cam = cv2.resize(cam, (224, 224))
#     return cam, score.item()

# def find_tumor_center(mask):
#     mask_uint8 = (mask * 255).astype(np.uint8)
#     contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if contours:
#         c = max(contours, key=cv2.contourArea)
#         M = cv2.moments(c)
#         if M["m00"] > 0:
#             return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
#     return None

# def generate_gradcam_overlay(image_path, confidence_threshold=0.7):
#     try:
#         model, gradients, activations = get_segmentation_model()

#         image = Image.open(image_path).convert("RGB")
#         img_tensor = transform_segment(image)
#         img_array = np.array(image.resize((224, 224)))

#         with torch.no_grad():
#             pred_mask = model(img_tensor.unsqueeze(0).to(device))
#             pred_mask_bin = (pred_mask > 0.5).float().squeeze().cpu().numpy()

#         cam, confidence = compute_gradcam(img_tensor, model, gradients, activations)
#         cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
#         overlay = cv2.addWeighted(img_array, 0.5, cam_heatmap, 0.5, 0)

#         center = find_tumor_center(pred_mask_bin)
#         if center and confidence > confidence_threshold:
#             cv2.circle(img_array, center, 56, (0, 255, 0), 2)
#             cv2.circle(overlay, center, 56, (0, 255, 0), 2)

#         filename = f"{uuid4().hex}_gradcam.png"
#         output_path = os.path.join("media", "gradcam", filename)
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#         cv2.imwrite(output_path, overlay)

#         torch.cuda.empty_cache()
#         return f"gradcam/{filename}", confidence

#     except Exception as e:
#         print(f"❌ GradCAM error: {e}")
#         return None, None

# INFER_URL = os.environ.get("NEUROVISTA_INFER_URL")  # set this in PythonAnywhere Web tab

# def infer_remote(image_file):
#     if not INFER_URL:
#         raise RuntimeError("NEUROVISTA_INFER_URL is not set")

#     # image_file is Django UploadedFile
#     image_file.seek(0)
#     files = {"file": (image_file.name, image_file.read(), image_file.content_type or "application/octet-stream")}
#     r = requests.post(INFER_URL, files=files, timeout=120)
#     r.raise_for_status()
#     return r.json()

# def decode_mask_png_base64(mask_b64: str) -> Image.Image:
#     raw = base64.b64decode(mask_b64)
#     return Image.open(io.BytesIO(raw)).convert("L")

import os
import base64
import io
import requests
from PIL import Image

INFER_URL = os.environ.get("NEUROVISTA_INFER_URL", "").strip()  # e.g. https://xxxx.hf.space/infer

class InferenceServiceError(RuntimeError):
    pass

def _call_remote_infer(uploaded_file, timeout=180):
    if not INFER_URL:
        raise InferenceServiceError("NEUROVISTA_INFER_URL is not set on the server.")

    uploaded_file.seek(0)
    files = {
        "file": (uploaded_file.name, uploaded_file.read(), uploaded_file.content_type or "application/octet-stream")
    }

    try:
        r = requests.post(INFER_URL, files=files, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise InferenceServiceError(f"Inference request failed: {e}")

def _b64_to_pil_png(b64_str: str) -> Image.Image:
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw))

# --- Your existing function names (so views.py doesn't break) ---

def detect_tumor(uploaded_file):
    data = _call_remote_infer(uploaded_file)
    return data.get("detection", {"has_tumor": None, "confidence": None})

def classify_tumor(uploaded_file):
    data = _call_remote_infer(uploaded_file)
    return data.get("classification", {"label": None, "confidence": None})

def segment_tumor(uploaded_file):
    data = _call_remote_infer(uploaded_file)
    mask_b64 = data.get("mask_png_base64")
    if not mask_b64:
        return None
    return _b64_to_pil_png(mask_b64).convert("L")  # PIL image mask

def generate_gradcam_overlay(uploaded_file):
    data = _call_remote_infer(uploaded_file)
    overlay_b64 = data.get("gradcam_png_base64")
    if not overlay_b64:
        return None
    return _b64_to_pil_png(overlay_b64).convert("RGB")
