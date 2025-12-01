import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Load the PoseNet model from TensorFlow Hub
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Provide real-time feedback
def provide_feedback(frame, keypoints):
    # Keypoints for shoulder and elbow joints
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_elbow = keypoints[7]
    right_elbow = keypoints[8]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    
    # Calculate arm angles for T-pose (target: ~180 degrees)
    left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    
    # Check alignment and give feedback
    feedback_text = ""
    if abs(left_arm_angle - 180) > 10:
        feedback_text += "Left arm not straight. "
    if abs(right_arm_angle - 180) > 10:
        feedback_text += "Right arm not straight. "
        
    # Display feedback
    if feedback_text:
        cv2.putText(frame, feedback_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Good Pose!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Function to process frame and detect pose
def detect_pose(frame, model):
    # Preprocess the frame
    input_image = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 192, 192)
    input_image = tf.cast(input_image, dtype=tf.int32)

    # Run inference
    outputs = model.signatures["serving_default"](input_image)
    keypoints = outputs['output_0'].numpy()[0, 0, :, :2]

    # Scale keypoints to the original frame size
    height, width, _ = frame.shape
    keypoints = np.array([[int(point[0] * width), int(point[1] * height)] for point in keypoints])

    return keypoints

# Main function to run the pose detection
def main():
    cap = cv2.VideoCapture(0)  # Start video capture
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect poses
        keypoints = detect_pose(frame, model)
        
        # Provide feedback based on keypoints
        provide_feedback(frame, keypoints)
        
        # Display the frame
        cv2.imshow('Yoga Pose Correction', frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    #dataset
# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("niharika41298/yoga-poses-dataset")

# print("Path to dataset files:", path)

# import cv2
# import numpy as np
# import tensorflow as tf
# import tensorflow_hub as hub

# # Load the PoseNet model from TensorFlow Hub
# model = hub.load("https://tfhub.dev/google/posenet/resnet50/1").signatures['serving_default']

# # Function to calculate angle between three points
# def calculate_angle(a, b, c):
#     a = np.array(a)
#     b = np.array(b)
#     c = np.array(c)
    
#     radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
#     angle = np.abs(radians * 180.0 / np.pi)
    
#     if angle > 180.0:
#         angle = 360 - angle
        
#     return angle

# # Provide real-time feedback
# def provide_feedback(frame, keypoints):
#     # Extract keypoints for shoulders, elbows, and wrists
#     left_shoulder = keypoints[5]
#     right_shoulder = keypoints[6]
#     left_elbow = keypoints[7]
#     right_elbow = keypoints[8]
#     left_wrist = keypoints[9]
#     right_wrist = keypoints[10]
    
#     # Calculate arm angles
#     left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
#     right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    
#     # Pose correction feedback
#     feedback_text = ""
#     if abs(left_arm_angle - 180) > 10:
#         feedback_text += "Left arm not straight. "
#     if abs(right_arm_angle - 180) > 10:
#         feedback_text += "Right arm not straight. "
        
#     # Display feedback
#     if feedback_text:
#         cv2.putText(frame, feedback_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#     else:
#         cv2.putText(frame, "Good Pose!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# # Function to process frame and detect pose using PoseNet
# def detect_pose(frame, model):
#     # Convert frame to tensor
#     image = tf.image.resize_with_pad(tf.convert_to_tensor(frame), 257, 257)
#     image = tf.cast(image, dtype=tf.float32)
#     image = image[tf.newaxis, ...]  # Add batch dimension

#     # Run PoseNet inference
#     outputs = model(tf.constant(image))
#     keypoints = outputs['output_0'].numpy()[0, :, :2]  # Extract keypoints (x, y)

#     # Scale keypoints to original frame size
#     height, width, _ = frame.shape
#     keypoints = np.array([[int(point[0] * width), int(point[1] * height)] for point in keypoints])

#     return keypoints

# # Main function to run the pose detection
# def main():
#     cap = cv2.VideoCapture(0)  # Start video capture
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Detect pose using PoseNet
#         keypoints = detect_pose(frame, model)
        
#         # Provide feedback based on keypoints
#         provide_feedback(frame, keypoints)
        
#         # Display the frame
#         cv2.imshow('Yoga Pose Correction - PoseNet', frame)
        
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
