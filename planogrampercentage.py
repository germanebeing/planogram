import torch
import cv2
import yaml
import numpy as np
from PIL import Image

def load_model(model_path):
    # Load the .pt model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    return model

def load_yaml(yaml_path):
    # Load product names from the YAML file
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data['names']

def convert_to_valid_jpg(image_path):
    """
    Converts the image to a valid JPG format if necessary.
    """
    try:
        with Image.open(image_path) as img:
            if img.format != 'JPEG':  # Check if it's not already a valid JPEG
                print(f"Converting {image_path} to a valid JPG format...")
                rgb_image = img.convert('RGB')  # Ensure RGB format
                new_path = image_path.replace(".png", ".jpg")
                rgb_image.save(new_path, format='JPEG')
                return new_path
    except Exception as e:
        print(f"Error converting image: {e}")
    return image_path

def infer_image(model, image_path):
    # Ensure the image is in a valid JPG format
    image_path = convert_to_valid_jpg(image_path)
    # Run the model inference on the image
    results = model(image_path)
    return results

def parse_detections(results, product_names):
    # Extract detections and map to product names
    detections = []
    print("Raw Results:", results.xyxy[0])  # Debug: Log raw results
    for *box, conf, cls in results.xyxy[0]:  # Extract bbox, confidence, and class
        cls_index = int(cls)
        if cls_index < len(product_names):  # Ensure index is within bounds
            label = product_names[cls_index]
            detections.append({
                'label': label,
                'bbox': box,  # x_min, y_min, x_max, y_max
                'confidence': conf.item()
            })
        else:
            print(f"Warning: Skipping invalid class index {cls_index}")
    # Sort detections left to right based on bbox x-coordinates
    detections = sorted(detections, key=lambda x: x['bbox'][0])
    return detections

def check_compliance(detections, planogram_list):
    # Extract product labels from detections
    detected_order = [det['label'] for det in detections]
    
    # Reduce to unique sequence (remove consecutive duplicates)
    detected_sequence = []
    for label in detected_order:
        if not detected_sequence or detected_sequence[-1] != label:
            detected_sequence.append(label)
    
    # Compare detected sequence to planogram list
    compliance_count = 0
    issues = []
    for idx, (plan_item, detected_item) in enumerate(zip(planogram_list, detected_sequence)):
        if plan_item == detected_item:
            compliance_count += 1
        else:
            issues.append(f"Position {idx+1}: Replace '{detected_item}' with '{plan_item}'")

    # Check for missing products
    if len(detected_sequence) < len(planogram_list):
        for idx in range(len(detected_sequence), len(planogram_list)):
            issues.append(f"Position {idx+1}: Missing '{planogram_list[idx]}'")

    compliance_percentage = (compliance_count / len(planogram_list)) * 100

    # Debugging: Log detected sequence and planogram list
    print("Detected Sequence:", detected_sequence)
    print("Planogram List:", planogram_list)
    print(f"Compliance Count: {compliance_count}")
    print(f"Compliance Percentage: {compliance_percentage:.2f}%")
    print(f"Issues: {issues}")
    
    return detected_sequence == planogram_list, compliance_percentage, issues

def annotate_image(image_path, detections, compliance, compliance_percentage, planogram_list, issues):
    # Load image
    image = cv2.imread(image_path)

    # Add space for annotations below the image
    height, width, _ = image.shape
    new_height = height + 200  # Add 200 pixels for text annotations
    annotated_image = np.zeros((new_height, width, 3), dtype=np.uint8)
    annotated_image[:height, :, :] = image  # Copy original image to the top

    # Draw bounding boxes and annotate compliance
    for det in detections:
        x_min, y_min, x_max, y_max = map(int, det['bbox'])
        color = (0, 255, 0) if det['label'] in planogram_list else (0, 0, 255)
        cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(annotated_image, det['label'], (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add compliance label and percentage
    label = "Shelf is Planogram Compliant" if compliance else "Not Compliant"
    cv2.putText(annotated_image, label, (10, height + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(annotated_image, f"Compliance: {compliance_percentage:.2f}%", (10, height + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Add issue annotations
    for i, issue in enumerate(issues):
        cv2.putText(annotated_image, issue, (10, height + 100 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return annotated_image

def main(image_path, model_path, yaml_path, planogram_list):
    # Load model and product names
    model = load_model(model_path)
    product_names = load_yaml(yaml_path)
    
    # Run inference
    results = infer_image(model, image_path)
    detections = parse_detections(results, product_names)
    
    # Check compliance
    compliance, compliance_percentage, issues = check_compliance(detections, planogram_list)
    
    # Annotate image
    annotated_image = annotate_image(image_path, detections, compliance, compliance_percentage, planogram_list, issues)
    
    # Display or save the result
    cv2.imshow("Planogram Compliance", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
planogram_list = [ 
    "Thumbs Up PET 750 ml",
    "Sprite PET 750 ml",
    "Thumbs Up PET 1000 ml",
    "Sprite PET 1000 ml"
]
image_path = "/Users/prithvi/Downloads/testplanogram.jpg"
model_path = "/Users/prithvi/Downloads/slmgrun4.pt"
yaml_path = "/Users/prithvi/Downloads/slmgrun4.yaml"

main(image_path, model_path, yaml_path, planogram_list)
