# Import necessary libraries
import cv2  # OpenCV for camera access and image processing
import pyttsx3 # For text-to-speech
import time # For adding delays if needed
import os # For path joining
import numpy as np # For numerical operations, especially with YOLO output

# --- Configuration ---
AI_NAME_LONG = "Demon Snow"
AI_NAME_SHORT = "Snow"
USE_GUI = True # Set to False if you want to run without the camera window

# --- DNN Model Configuration (YOLOv3-tiny) ---
MODEL_DIR = "dnn_model"
MODEL_CONFIG = os.path.join(MODEL_DIR, "yolov3-tiny.cfg") # YOLO config file
MODEL_WEIGHTS = os.path.join(MODEL_DIR, "yolov3-tiny.weights") # YOLO weights file
CLASS_LABELS_FILE = os.path.join(MODEL_DIR, "coco.names")

CONFIDENCE_THRESHOLD = 0.5 # Minimum confidence to consider a detection
NMS_THRESHOLD = 0.4        # Non-Maximum Suppression threshold (for filtering overlapping boxes)
DNN_INPUT_SIZE = (416, 416) # Input size for YOLOv3-tiny

# Load class names
try:
    with open(CLASS_LABELS_FILE, 'rt') as f:
        CLASS_NAMES = f.read().rstrip('\n').split('\n')
except FileNotFoundError:
    print(f"Error: Class labels file '{CLASS_LABELS_FILE}' not found. Make sure it's in the '{MODEL_DIR}' directory.")
    CLASS_NAMES = []
    # exit() 

# Load the DNN model (YOLO)
try:
    if os.path.exists(MODEL_CONFIG) and os.path.exists(MODEL_WEIGHTS):
        net = cv2.dnn.readNetFromDarknet(MODEL_CONFIG, MODEL_WEIGHTS)
        # Example: Set preferable backend and target if you have an NVIDIA GPU and OpenCV built with CUDA support
        # if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        #     net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        #     net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        #     print("Attempting to use CUDA backend for DNN.")
        # else:
        #     print("CUDA not available or no CUDA-enabled GPU found. Using CPU for DNN.")
        print("YOLOv3-tiny DNN model loaded successfully.")
    else:
        print(f"Error: YOLO Model files not found. Check paths: \nConfig: {MODEL_CONFIG}\nWeights: {MODEL_WEIGHTS}")
        net = None
        # exit()
except cv2.error as e:
    print(f"OpenCV Error loading YOLO DNN model: {e}")
    net = None
except Exception as e:
    print(f"General Error loading YOLO DNN model: {e}")
    net = None


# Initialize Text-to-Speech engine
try:
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    if voices and len(voices) > 1: # Check if voices list is not empty and has more than one voice
        try:
            engine.setProperty('voice', voices[1].id) 
            # print(f"Attempting to use voice: {voices[1].name}") 
        except Exception: # Fallback to the first voice if setting the second one fails
            if voices: engine.setProperty('voice', voices[0].id)
    elif voices: # If only one voice is available
        engine.setProperty('voice', voices[0].id)
    # else: No voices found, engine will use default or may not work.
    
    engine.setProperty('rate', 160)
    engine.setProperty('volume', 0.9)
except Exception as e:
    print(f"Error initializing TTS engine: {e}")
    engine = None

# --- Core Functions ---

def speak(text, use_short_name=True):
    ai_speaker_name = AI_NAME_SHORT if use_short_name else AI_NAME_LONG
    full_text_console = f"{ai_speaker_name}: {text}"
    print(full_text_console)
    if engine:
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"Error during speech: {e}")
    else:
        print("TTS engine not available for speech.")

def initialize_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        speak("Error: Could not open camera.", use_short_name=False)
        return None
    speak("Camera initialized successfully.", use_short_name=False)
    return cap

def get_output_layers(net):
    """Get the names of the output layers from the network"""
    layer_names = net.getLayerNames()
    try:
        output_layer_indices = net.getUnconnectedOutLayers()
        if isinstance(output_layer_indices, np.ndarray) and output_layer_indices.ndim == 2:
            output_layer_indices = output_layer_indices.flatten()
        return [layer_names[i - 1] for i in output_layer_indices]
    except AttributeError: # Handle cases where getUnconnectedOutLayers might return a list of lists of ints
         return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    except Exception as e:
         print(f"Error getting output layers: {e}")
         # Fallback or re-raise, depending on how critical this is.
         # For now, let's assume it might be an older OpenCV and try the common old way if the above fails.
         # This part might need adjustment based on the exact OpenCV version if errors persist here.
         try:
            return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
         except Exception as e_inner:
            print(f"Fallback for get_output_layers also failed: {e_inner}")
            return [] # Return empty list to prevent crash, detection will fail.


def detect_objects_in_frame(frame):
    """
    Detects objects in the frame using the loaded YOLOv3-tiny model.
    Returns a list of detected object names, confidences, and their bounding boxes.
    """
    detected_objects_info = [] 
    
    if net is None or not CLASS_NAMES:
        return detected_objects_info

    if frame is None:
        print("Warning: Input frame for detection is None.")
        return detected_objects_info

    frame_height, frame_width = frame.shape[:2]
    
    try:
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, DNN_INPUT_SIZE, swapRB=True, crop=False)
        net.setInput(blob)
        
        output_layers = get_output_layers(net)
        if not output_layers: # If getting output layers failed
            print("Error: Could not determine output layers for YOLO. Detection aborted.")
            return detected_objects_info
            
        layer_outputs = net.forward(output_layers)
        
        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:] 
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > CONFIDENCE_THRESHOLD:
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    width = int(detection[2] * frame_width)
                    height = int(detection[3] * frame_height)
                    
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)
                    
                    boxes.append([x, y, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

        if len(indices) > 0:
            if isinstance(indices, np.ndarray) and indices.ndim == 2:
                processed_indices = indices.flatten()
            else:
                processed_indices = indices # Assuming it's already flat or a simple list/tuple

            for i in processed_indices:
                box = boxes[i]
                x, y, w, h = box[0], box[1], box[2], box[3]
                
                class_name = "Unknown"
                if class_ids[i] < len(CLASS_NAMES): # Check index bounds
                    class_name = CLASS_NAMES[class_ids[i]]
                else:
                    print(f"Warning: class_id {class_ids[i]} out of bounds for CLASS_NAMES (len {len(CLASS_NAMES)})")

                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(frame_width -1 , x + w), min(frame_height -1, y + h)

                if x2 > x1 and y2 > y1 : # Ensure valid box after clamping
                    detected_objects_info.append((class_name, confidences[i], (x1, y1, x2-x1, y2-y1))) 
    
    except cv2.error as e:
        print(f"OpenCV error during YOLO detection: {e}")
    except Exception as e:
        print(f"General error during YOLO detection: {e}")
        
    return detected_objects_info


def main_loop():
    global USE_GUI

    speak(f"Initializing {AI_NAME_LONG} systems. Please stand by.", use_short_name=False)
    
    if net is None:
        speak("Warning: Object detection model (YOLO) could not be loaded. Detection will be disabled.", use_short_name=True)
    if not CLASS_NAMES:
        speak("Warning: Class names not loaded. Object names might be incorrect or missing.", use_short_name=True)

    cap = initialize_camera()

    if not cap:
        speak("Exiting due to camera initialization failure.", use_short_name=False)
        return

    speak("Systems online. I am now observing.", use_short_name=True)

    last_speech_time = time.time()
    speech_interval = 7 
    # More persistent memory for announced objects
    announced_objects_memory = set() 
    frames_with_no_detections_streak = 0
    # Reset memory if nothing seen for this many speech intervals
    MAX_NO_DETECTION_STREAK_FOR_MEMORY_RESET = 2 


    gui_initialized_successfully = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                speak("Error: Can't receive frame (stream end?). Exiting ...", use_short_name=True)
                break

            detected_objects_info = detect_objects_in_frame(frame.copy()) 
            
            current_time = time.time()
            
            if current_time - last_speech_time > speech_interval:
                current_frame_object_names = set(name for name, _, _ in detected_objects_info)

                if not current_frame_object_names:
                    frames_with_no_detections_streak += 1
                else:
                    # If objects are detected, reset the streak
                    frames_with_no_detections_streak = 0 

                # If nothing has been detected for a while, clear memory
                if frames_with_no_detections_streak >= MAX_NO_DETECTION_STREAK_FOR_MEMORY_RESET:
                    if announced_objects_memory: # Only speak if memory wasn't already empty
                        speak("The scene appears to be clear now.")
                    announced_objects_memory.clear()
                    frames_with_no_detections_streak = 0 # Reset streak after clearing

                # Determine which of the currently seen objects are new
                objects_to_announce = current_frame_object_names - announced_objects_memory
                
                if objects_to_announce:
                    speakable_names = list(objects_to_announce)
                    if len(speakable_names) == 1:
                        speak(f"I now see a {speakable_names[0]}.")
                    elif len(speakable_names) > 1:
                        if len(speakable_names) > 2:
                            items_speech = "a " + ", a ".join(speakable_names[:-1]) + f", and a {speakable_names[-1]}"
                        else: # Exactly two items
                            items_speech = f"a {speakable_names[0]} and a {speakable_names[1]}"
                        speak(f"I can also see {items_speech}.")
                    
                    # Add newly announced objects to the persistent memory
                    announced_objects_memory.update(objects_to_announce)
                
                last_speech_time = current_time
            # End of speech block

            if USE_GUI:
                try:
                    # Draw bounding boxes on the frame for all currently detected objects
                    for name, confidence, box in detected_objects_info:
                        x, y, w, h = box
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        label = f"{name}: {confidence:.2f}"
                        cv2.putText(frame, label, (x, y - 10 if y - 10 > 10 else y + 15), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    cv2.imshow(f'{AI_NAME_SHORT} Vision (YOLO) - Press Q to Quit', frame)
                    gui_initialized_successfully = True
                except cv2.error as e:
                    if "The function is not implemented" in str(e):
                        if not gui_initialized_successfully: # Show detailed warning only once
                            print("\n***************************************************************************")
                            print("OpenCV GUI Error: Cannot display camera window.")
                            print("Please ensure OpenCV is installed correctly with GUI support (e.g., pip install opencv-python).")
                            print("If issues persist, your system might lack underlying GUI libraries for OpenCV.")
                            print("Setting USE_GUI = False to continue without camera view.")
                            print("***************************************************************************\n")
                            speak("Warning: I cannot display the camera feed. Disabling GUI.", use_short_name=True)
                        USE_GUI = False # Disable further GUI attempts
                    else: # Other OpenCV errors related to imshow
                        print(f"An unexpected OpenCV error occurred with imshow: {e}")
                        speak("An error occurred with the display window. Disabling GUI.", use_short_name=True)
                        USE_GUI = False 
            
            # Key press handling
            if gui_initialized_successfully or USE_GUI : # If GUI is supposed to be active
                key_press = cv2.waitKey(1) & 0xFF
                if key_press == ord('q'):
                    speak("Deactivating. Farewell.", use_short_name=True)
                    break
            else: # If GUI is off, allow console interrupt or just loop
                time.sleep(0.05) # Small delay to prevent tight loop if no GUI and no waitKey

    except KeyboardInterrupt:
        speak("Deactivation requested via Keyboard Interrupt. Farewell.", use_short_name=True)
    finally:
        if cap:
            cap.release()
        if gui_initialized_successfully: # Only destroy windows if they were shown
            try:
                cv2.destroyAllWindows()
            except cv2.error as e:
                print(f"Notice: cv2.destroyAllWindows() also encountered an issue: {e}")
        speak(f"{AI_NAME_LONG} systems offline.", use_short_name=False)

if __name__ == '__main__':
    # Initial check for model files before starting anything else
    if not all([os.path.exists(MODEL_CONFIG), os.path.exists(MODEL_WEIGHTS), os.path.exists(CLASS_LABELS_FILE)]):
        print("\n--- CRITICAL ERROR ---")
        print("One or more YOLO DNN model files are missing from the 'dnn_model' directory:")
        if not os.path.exists(MODEL_CONFIG): print(f" - Missing: {MODEL_CONFIG}")
        if not os.path.exists(MODEL_WEIGHTS): print(f" - Missing: {MODEL_WEIGHTS}")
        if not os.path.exists(CLASS_LABELS_FILE): print(f" - Missing: {CLASS_LABELS_FILE}")
        print("Please download them and place them correctly. Exiting.")
    else:
        speak(f"Hello! I am {AI_NAME_LONG}, but you can call me {AI_NAME_SHORT}.", use_short_name=False)
        main_loop()
