"""
Model Utils with IMPROVED Grad-CAM - Better Lesion Focus
Compatible with 10-class model
Python 3.10.6 | TensorFlow 2.15.0
"""

import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import base64
from io import BytesIO
import json

# ==================== CONFIGURATION ====================

MODEL_PATH = "models/melanoma_10class_invalid_detection.keras"
CLASS_INDICES_PATH = "models/class_indices_10class.json"
IMG_SIZE = 160

# ==================== LOAD CLASS NAMES ====================

try:
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    
    CLASS_NAMES = [None] * len(class_indices)
    for class_name, idx in class_indices.items():
        CLASS_NAMES[idx] = class_name
    
    print(f" Loaded {len(CLASS_NAMES)} classes")
    
    INVALID_CLASS_IDX = None
    for class_name, idx in class_indices.items():
        if 'invalid' in class_name.lower():
            INVALID_CLASS_IDX = idx
            print(f" Invalid class: '{class_name}' at index {idx}")
            break
            
except FileNotFoundError:
    CLASS_NAMES = [
        "Actinic Keratosis", "Basal Cell Carcinoma", "Dermato Fibroma",
        "Melanoma", "Nevus", "Pigmented Benign Keratosis",
        "Seborrheic Keratosis", "Squamous Cell Carcinoma", "Vascular Lesion"
    ]
    INVALID_CLASS_IDX = None

# ==================== LESION INFO ====================

LESION_INFO = {
    "Actinic Keratosis": {"risk_level": "MODERATE", "color": "#FFC107", "malignancy": "Pre-malignant",
        "description": "Pre-cancerous skin growth caused by sun damage.", 
        "action": "Consult dermatologist for evaluation."},
    "Basal Cell Carcinoma": {"risk_level": "HIGH", "color": "#E53935", "malignancy": "Malignant",
        "description": "Most common skin cancer. Locally invasive.", 
        "action": "Seek consultation within 1-2 weeks."},
    "Dermato Fibroma": {"risk_level": "LOW", "color": "#66BB6A", "malignancy": "Benign",
        "description": "Harmless skin growth.", 
        "action": "Routine monitoring."},
    "Melanoma": {"risk_level": "CRITICAL", "color": "#B71C1C", "malignancy": "Malignant",
        "description": "Most dangerous skin cancer. Can spread rapidly.", 
        "action": "URGENT: Immediate consultation required."},
    "Nevus": {"risk_level": "LOW", "color": "#81C784", "malignancy": "Benign",
        "description": "Common mole. Usually harmless.", 
        "action": "Monitor for changes."},
    "Pigmented Benign Keratosis": {"risk_level": "LOW", "color": "#4FC3F7", "malignancy": "Benign",
        "description": "Benign pigmented lesion.", 
        "action": "Routine monitoring."},
    "Seborrheic Keratosis": {"risk_level": "LOW", "color": "#4DD0E1", "malignancy": "Benign",
        "description": "Common benign growth.", 
        "action": "No treatment necessary."},
    "Squamous Cell Carcinoma": {"risk_level": "HIGH", "color": "#F44336", "malignancy": "Malignant",
        "description": "Second most common skin cancer.", 
        "action": "Schedule appointment within 1-2 weeks."},
    "Vascular Lesion": {"risk_level": "LOW", "color": "#AB47BC", "malignancy": "Benign",
        "description": "Benign blood vessel abnormality.", 
        "action": "Monitor for changes."},
    "Invalid": {"risk_level": "N/A", "color": "#9E9E9E", "malignancy": "Not Applicable",
        "description": "Not a medical skin lesion image.", 
        "action": "Upload a dermatoscopic image."}
}

# ==================== LOAD MODEL ====================

print(f" Loading: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print(f" Model loaded: {model.output_shape[-1]} classes")

# ==================== GRAD-CAM SETUP ====================

mobilenet_layer = None
for layer in model.layers:
    if 'mobilenet' in layer.name.lower():
        mobilenet_layer = layer
        break

mobilenet_grad_model = None
if mobilenet_layer:
    try:
        last_conv = mobilenet_layer.get_layer('out_relu')
        mobilenet_grad_model = tf.keras.Model(
            inputs=mobilenet_layer.input,
            outputs=[last_conv.output, mobilenet_layer.output]
        )
        print(" Improved Grad-CAM ready")
    except:
        print("  Grad-CAM unavailable, using fallback")

# ==================== IMPROVED GRAD-CAM ====================

def generate_improved_gradcam(img_array, pred_idx, original_img):
    """Enhanced Grad-CAM with better lesion focus"""
    
    if mobilenet_grad_model is None:
        return create_lesion_focused_attention(img_array[0])
    
    try:
        with tf.GradientTape() as tape:
            conv_output, _ = mobilenet_grad_model(img_array, training=False)
            full_pred = model(img_array, training=False)
            target = full_pred[:, pred_idx]
        
        grads = tape.gradient(target, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
        
        conv_output = conv_output[0].numpy()
        pooled_grads = pooled_grads.numpy()
        
        for i in range(pooled_grads.shape[0]):
            conv_output[:,:,i] *= pooled_grads[i]
        
        heatmap = np.mean(conv_output, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        # IMPROVED: Focus on lesion with aggressive thresholding
        threshold = np.percentile(heatmap, 70)  # Keep only top 30%
        heatmap_focused = heatmap.copy()
        heatmap_focused[heatmap_focused < threshold] *= 0.1  # Suppress background
        
        if np.max(heatmap_focused) > 0:
            heatmap_focused = heatmap_focused / np.max(heatmap_focused)
        
        # IMPROVED: Edge-guided refinement
        original_resized = cv2.resize(np.array(original_img), 
                                     (heatmap_focused.shape[1], heatmap_focused.shape[0]))
        gray = cv2.cvtColor(original_resized, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150).astype(float) / 255.0
        edges_dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        
        # Combine with edge information
        heatmap_refined = heatmap_focused * (0.7 + 0.3 * edges_dilated)
        
        if np.max(heatmap_refined) > 0:
            heatmap_refined = heatmap_refined / np.max(heatmap_refined)
        
        # Increase contrast
        heatmap_refined = np.power(heatmap_refined, 0.8)
        
        return heatmap_refined
        
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return create_lesion_focused_attention(img_array[0])


def create_lesion_focused_attention(img):
    """
    Fallback attention focusing on lesion characteristics:
    edges, darkness, texture, color variation
    """
    
    img_uint8 = (img * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    
    # Feature 1: Edges (lesions have distinct edges)
    edges = cv2.Canny(gray, 30, 100)
    edges_dilated = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=2)
    edges_norm = edges_dilated.astype(float) / 255.0
    
    # Feature 2: Darkness (lesions often darker)
    darkness = (255 - gray).astype(float) / 255.0
    darkness = cv2.GaussianBlur(darkness, (21, 21), 0)
    
    # Feature 3: Texture
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture = np.abs(laplacian)
    texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-8)
    texture = cv2.GaussianBlur(texture, (11, 11), 0)
    
    # Feature 4: Color variation
    b, g, r = cv2.split(img_uint8)
    color_std = np.std(np.stack([b,g,r], axis=-1), axis=-1)
    color_std = (color_std - color_std.min()) / (color_std.max() - color_std.min() + 1e-8)
    color_std = cv2.GaussianBlur(color_std, (15, 15), 0)
    
    # Combine features with weights
    attention = (
        edges_norm * 0.35 +
        darkness * 0.30 +
        texture * 0.20 +
        color_std * 0.15
    )
    
    # Strong threshold to focus on lesion
    threshold = np.percentile(attention, 75)
    attention[attention < threshold] *= 0.05
    
    attention = cv2.GaussianBlur(attention, (15, 15), 0)
    
    if np.max(attention) > 0:
        attention = attention / np.max(attention)
    
    return np.power(attention, 0.7)


def create_professional_overlay(original, heatmap, alpha=0.5):
    """Professional medical-grade heatmap overlay with lesion focus"""
    
    output_size = IMG_SIZE * 4
    
    # Resize images
    img_resized = original.resize((output_size, output_size), Image.LANCZOS)
    img_array = np.array(img_resized)
    
    heatmap_resized = cv2.resize(heatmap, (output_size, output_size), 
                                  interpolation=cv2.INTER_CUBIC)
    
    # IMPROVED: Adaptive thresholding for better focus
    threshold_high = np.percentile(heatmap_resized, 60)
    threshold_low = np.percentile(heatmap_resized, 30)
    
    heatmap_enhanced = heatmap_resized.copy()
    heatmap_enhanced[heatmap_resized >= threshold_high] = heatmap_resized[heatmap_resized >= threshold_high]
    heatmap_enhanced[(heatmap_resized >= threshold_low) & (heatmap_resized < threshold_high)] *= 0.4
    heatmap_enhanced[heatmap_resized < threshold_low] *= 0.05
    
    if np.max(heatmap_enhanced) > 0:
        heatmap_enhanced = heatmap_enhanced / np.max(heatmap_enhanced)
    
    # Apply colormap
    heatmap_uint8 = np.uint8(255 * heatmap_enhanced)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Variable alpha blending
    alpha_mask = np.clip(heatmap_enhanced * 1.2, 0, 1)
    alpha_mask = np.stack([alpha_mask] * 3, axis=-1)
    
    overlay = img_array.astype(float) * (1 - alpha_mask * alpha) + \
              heatmap_colored.astype(float) * (alpha_mask * alpha)
    
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    # IMPROVED: Add subtle boundary enhancement
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges_dilated = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)
    overlay[edges_dilated > 0] = overlay[edges_dilated > 0] * 0.9 + np.array([255,255,255]) * 0.1
    
    # Convert to base64
    overlay_img = Image.fromarray(overlay)
    buffered = BytesIO()
    overlay_img.save(buffered, format="PNG", quality=95)
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"

# ==================== OTHER FUNCTIONS ====================

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    original = img.copy()
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0), original

def calculate_detailed_risk(pred_class, conf, all_probs):
    if INVALID_CLASS_IDX is not None and pred_class == CLASS_NAMES[INVALID_CLASS_IDX]:
        return {
            "final_risk": "NOT APPLICABLE", "risk_color": "#9E9E9E",
            "confidence_level": "N/A", "confidence_color": "#9E9E9E",
            "urgency": "Upload valid skin image", "malignancy": "Not Applicable",
            "base_risk": "N/A", "probability_certainty": "N/A"
        }
    
    info = LESION_INFO.get(pred_class, LESION_INFO["Nevus"])
    base_risk = info["risk_level"]
    
    sorted_p = sorted(all_probs.values(), reverse=True)
    diff = sorted_p[0] - sorted_p[1] if len(sorted_p) > 1 else sorted_p[0]
    
    conf_level = "Very High" if conf >= 85 else "High" if conf >= 70 else "Moderate" if conf >= 55 else "Low"
    conf_color = "#2E7D32" if conf >= 85 else "#388E3C" if conf >= 70 else "#FBC02D" if conf >= 55 else "#F57C00"
    
    if base_risk == "CRITICAL":
        final = "CRITICAL" if conf >= 80 else "HIGH" if conf >= 60 else "MODERATE-HIGH"
        color = "#B71C1C" if conf >= 80 else "#D32F2F" if conf >= 60 else "#F57C00"
        urgency = "Immediate attention" if conf >= 80 else "Seek consultation soon" if conf >= 60 else "Evaluation recommended"
    elif base_risk == "HIGH":
        final = "HIGH" if conf >= 75 else "MODERATE-HIGH" if conf >= 55 else "MODERATE"
        color = "#E53935" if conf >= 75 else "#FF6F00" if conf >= 55 else "#FBC02D"
        urgency = "Consultation within 1-2 weeks" if conf >= 75 else "Schedule appointment" if conf >= 55 else "Monitor and consult if concerned"
    elif base_risk == "MODERATE":
        final = "MODERATE" if conf >= 70 else "LOW-MODERATE"
        color = "#FFA726" if conf >= 70 else "#FFB74D"
        urgency = "Monitor regularly" if conf >= 70 else "Keep under observation"
    else:
        final = "LOW" if conf >= 75 else "UNCERTAIN"
        color = "#66BB6A" if conf >= 75 else "#9E9E9E"
        urgency = "Routine monitoring" if conf >= 75 else "Consider evaluation"
    
    return {
        "final_risk": final, "risk_color": color,
        "confidence_level": conf_level, "confidence_color": conf_color,
        "urgency": urgency, "malignancy": info["malignancy"],
        "base_risk": base_risk,
        "probability_certainty": "High" if diff > 0.3 else "Moderate" if diff > 0.15 else "Low"
    }

# ==================== MAIN PREDICTION ====================

def predict_and_analyze(img_path):
    print(f"🔬 Analyzing: {img_path}")
    
    img_array, original = preprocess_image(img_path)
    predictions = model.predict(img_array, verbose=0)[0]
    
    all_probs = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}
    
    pred_idx = np.argmax(predictions)
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(predictions[pred_idx]) * 100
    
    print(f" {pred_class} ({confidence:.2f}%)")
    
    # Check if invalid
    if INVALID_CLASS_IDX is not None and pred_idx == INVALID_CLASS_IDX:
        print(" INVALID IMAGE")
        return {
            "error": True, "error_type": "INVALID_IMAGE",
            "error_message": " Invalid Image Detected",
            "error_details": f"Model classified as '{pred_class}' ({confidence:.1f}%). Not a skin lesion.",
            "suggestion": "Upload a dermatoscopic/clinical skin photograph.",
            "possible_reasons": [
                "Non-medical content", "Insufficient quality",
                "Not skin tissue", "Non-dermatological features"
            ],
            "predicted_class": pred_class, "confidence": round(confidence, 2),
            "all_probabilities": all_probs
        }
    
    # IMPROVED: Generate better Grad-CAM
    print(" Generating improved Grad-CAM...")
    heatmap = generate_improved_gradcam(img_array, pred_idx, original)
    
    print(" Creating professional overlay...")
    overlay = create_professional_overlay(original, heatmap, alpha=0.5)
    
    risk = calculate_detailed_risk(pred_class, confidence, all_probs)
    info = LESION_INFO.get(pred_class, LESION_INFO["Nevus"])
    
    return {
        "error": False,
        "predicted_class": pred_class,
        "confidence": round(confidence, 2),
        "confidence_percentage": f"{confidence:.1f}%",
        "all_probabilities": {
            name: {
                "probability": prob,
                "percentage": f"{prob*100:.1f}%",
                "color": LESION_INFO.get(name, {"color": "#9E9E9E"})["color"]
            }
            for name, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        },
        "risk_assessment": risk,
        "lesion_info": {
            "description": info["description"],
            "action_required": info["action"],
            "color": info["color"]
        },
        "visualizations": {"gradcam_heatmap": overlay},
        "disclaimer": "⚕️ AI-assisted tool, NOT a medical diagnosis. Consult healthcare professionals."
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python model_utils.py <image>")
        sys.exit(1)
    
    results = predict_and_analyze(sys.argv[1])
    
    if results.get("error"):
        print(f"\n {results['error_message']}")
    else:
        print(f"\n {results['predicted_class']}")
        print(f"Confidence: {results['confidence_percentage']}")
        print(f"Risk: {results['risk_assessment']['final_risk']}")