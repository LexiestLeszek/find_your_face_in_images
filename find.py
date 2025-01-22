from deepface import DeepFace
import os
import cv2

def find_my_face(reference_img_path, target_folder_path):
    # Verify reference image
    if not os.path.exists(reference_img_path):
        raise FileNotFoundError(f"Reference image not found: {reference_img_path}")
    
    # Configure settings :cite[7]
    model_name = "Facenet"  # High-accuracy model (98.4% measured score)
    detector_backend = "retinaface"  # Best detection accuracy
    distance_metric = "cosine"
    threshold = 0.4  # Lower = stricter matching
    
    # Get reference embedding
    reference_embedding = DeepFace.represent(
        img_path=reference_img_path,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=False
    )[0]["embedding"]

    # Scan target folder
    matches = []
    for root, _, files in os.walk(target_folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                try:
                    # Compare faces
                    result = DeepFace.verify(
                        img1_path=reference_img_path,
                        img2_path=img_path,
                        model_name=model_name,
                        detector_backend=detector_backend,
                        distance_metric=distance_metric,
                        enforce_detection=False
                    )
                    
                    if result["verified"] and result["distance"] < threshold:
                        matches.append({
                            "path": img_path,
                            "similarity": 1 - result["distance"],
                            "face_location": result["facial_areas"]["img2"]
                        })
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")

    return matches

if __name__ == "__main__":
    reference_path = "original.jpg"
    target_folder = "./search_imgs"
    
    matches = find_my_face(reference_path, target_folder)
    
    print(f"\nFound {len(matches)} matching images:")
    for match in matches:
        print(f"- {match['path']} (Similarity: {match['similarity']:.2%})")
    
    
