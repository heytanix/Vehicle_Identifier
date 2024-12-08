import cv2
import torch
import numpy as np
import torchvision

def get_available_cameras():
    """Get list of available camera devices and their names"""
    camera_list = []
    # Try cameras starting from index 0
    for i in range(10):  # Check first 10 indexes
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Get camera name/description
            name = f"Camera {i}"
            try:
                # Try to get camera name from backend (may not work on all systems)
                name = cap.getBackendName() + f" - Camera {i}"
            except:
                pass
            camera_list.append((i, name))
            cap.release()
    return camera_list

def track_cars():
    # Get available cameras
    cameras = get_available_cameras()
    if not cameras:
        print("No cameras found!")
        return
        
    # Display available cameras
    print("\nAvailable cameras:")
    for idx, (cam_id, cam_name) in enumerate(cameras):
        print(f"{idx + 1}. {cam_name}")
    
    # Let user choose camera
    while True:
        try:
            choice = int(input("\nSelect camera number (1-{}): ".format(len(cameras)))) - 1
            if 0 <= choice < len(cameras):
                camera_id = cameras[choice][0]
                break
            print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

    # Initialize YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5s model
    model.eval()
    
    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Initialize video capture with selected camera
    cap = cv2.VideoCapture(camera_id)
    print(f"\nUsing camera: {cameras[choice][1]}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert frame to RGB (YOLOv5 expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get predictions
        results = model(frame_rgb)
        
        # Filter for cars (class 2 in COCO dataset)
        cars = results.xyxy[0][results.xyxy[0][:, 5] == 2].cpu().numpy()
        
        # Draw bounding boxes for cars with confidence > 0.5
        for det in cars:
            x1, y1, x2, y2, conf, _ = det
            if conf > 0.5:
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'Car: {conf:.2f}', (int(x1), int(y1)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Car Tracking', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    track_cars()
