# Import necessary libraries
import cv2  # OpenCV for camera access and image processing
import pyttsx3 # For text-to-speech
import time # For adding delays if needed
import os # For path joining
import numpy as np # For numerical operations, especially with YOLO output

# --- Configuration ---
AI_NAME_LONG = "Deamon Snow"
AI_NAME_SHORT = "Snow"
USE_GUI = True 

# --- pyttsx3 Configuration ---
try:
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    if voices and len(voices) > 1:
        try:
            engine.setProperty('voice', voices[1].id) 
        except Exception: 
            if voices: engine.setProperty('voice', voices[0].id) 
    elif voices: 
        engine.setProperty('voice', voices[0].id)
    engine.setProperty('rate', 160)
    engine.setProperty('volume', 0.9)
    print("pyttsx3 engine initialized.")
except Exception as e:
    print(f"Error initializing pyttsx3 engine: {e}")
    engine = None

# --- DNN Model Configuration (YOLOv3-tiny) ---
MODEL_DIR = "dnn_model"
MODEL_CONFIG = os.path.join(MODEL_DIR, "yolov3-tiny.cfg")
MODEL_WEIGHTS = os.path.join(MODEL_DIR, "yolov3-tiny.weights")
CLASS_LABELS_FILE = os.path.join(MODEL_DIR, "coco.names")

CONFIDENCE_THRESHOLD = 0.5 
NMS_THRESHOLD = 0.4        
DNN_INPUT_SIZE = (416, 416) 

# --- Tracker Configuration ---
TRACKER_TYPE = "CSRT" 
IOU_THRESHOLD_FOR_NEW_TRACK = 0.3 # If IoU is less than this with existing tracks, consider it new
MAX_TRACKERS = 10 # Limit the number of concurrent trackers for performance
MAX_FRAMES_SINCE_SEEN_THRESHOLD = 3 # Number of DETECTION CYCLES before removing a stale track

# Load class names
try:
    with open(CLASS_LABELS_FILE, 'rt') as f:
        CLASS_NAMES = f.read().rstrip('\n').split('\n')
except FileNotFoundError:
    print(f"Error: Class labels file '{CLASS_LABELS_FILE}' not found.")
    CLASS_NAMES = []

# Load the DNN model (YOLO)
try:
    if os.path.exists(MODEL_CONFIG) and os.path.exists(MODEL_WEIGHTS):
        net = cv2.dnn.readNetFromDarknet(MODEL_CONFIG, MODEL_WEIGHTS)
        print("YOLOv3-tiny DNN model loaded successfully.")
    else:
        print(f"Error: YOLO Model files not found.")
        net = None
except cv2.error as e:
    print(f"OpenCV Error loading YOLO DNN model: {e}")
    net = None
except Exception as e:
    print(f"General Error loading YOLO DNN model: {e}")
    net = None

# --- Helper Functions ---
def speak(text, use_short_name=True):
    ai_speaker_name = AI_NAME_SHORT if use_short_name else AI_NAME_LONG
    full_text_console = f"{ai_speaker_name}: {text}"
    print(full_text_console)
    if engine:
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"Error during pyttsx3 speech: {e}")
    else:
        print("pyttsx3 engine not initialized. Cannot generate speech.")

def initialize_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        speak("Error: Could not open camera.", False)
        return None
    speak("Camera initialized successfully.", False)
    return cap

def get_output_layers(net_model):
    layer_names = net_model.getLayerNames()
    try:
        output_layer_indices = net_model.getUnconnectedOutLayers()
        if isinstance(output_layer_indices, np.ndarray) and output_layer_indices.ndim == 2:
            output_layer_indices = output_layer_indices.flatten()
        return [layer_names[i - 1] for i in output_layer_indices]
    except AttributeError: 
         return [layer_names[i[0] - 1] for i in net_model.getUnconnectedOutLayers()]
    except Exception as e:
         print(f"Error getting output layers: {e}")
         return []

def create_tracker_instance(tracker_type_str): # Renamed from create_tracker to avoid conflict
    tracker = None
    # print(f"Attempting to create tracker: {tracker_type_str}") # Less verbose
    try:
        if tracker_type_str == "CSRT":
            if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
                tracker = cv2.legacy.TrackerCSRT_create()
            elif hasattr(cv2, 'TrackerCSRT_create'): 
                tracker = cv2.TrackerCSRT_create()
        # Add other tracker types if needed
        if tracker is None:
            print(f"Warning: Could not create tracker of type {tracker_type_str}.")
    except Exception as e:
        print(f"Error creating tracker '{tracker_type_str}': {e}")
    return tracker

def calculate_iou(boxA, boxB):
    # box format: (x, y, w, h)
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def detect_objects_in_frame(frame_to_detect):
    detected_objects_info = [] 
    if net is None or not CLASS_NAMES or frame_to_detect is None:
        return detected_objects_info
    frame_height, frame_width = frame_to_detect.shape[:2]
    try:
        blob = cv2.dnn.blobFromImage(frame_to_detect, 1/255.0, DNN_INPUT_SIZE, swapRB=True, crop=False)
        net.setInput(blob)
        output_layers = get_output_layers(net)
        if not output_layers: return detected_objects_info
        layer_outputs = net.forward(output_layers)
        
        boxes, confidences, class_ids = [], [], []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:] 
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > CONFIDENCE_THRESHOLD:
                    center_x, center_y = int(detection[0] * frame_width), int(detection[1] * frame_height)
                    w, h = int(detection[2] * frame_width), int(detection[3] * frame_height)
                    x, y = int(center_x - w / 2), int(center_y - h / 2)
                    boxes.append([x, y, w, h]); confidences.append(float(confidence)); class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        if len(indices) > 0:
            processed_indices = indices.flatten() if isinstance(indices, np.ndarray) and indices.ndim == 2 else indices
            for i in processed_indices:
                box = boxes[i]
                x_coord, y_coord, w_val, h_val = box[0], box[1], box[2], box[3]
                class_name = CLASS_NAMES[class_ids[i]] if class_ids[i] < len(CLASS_NAMES) else "Unknown"
                x1, y1 = max(0, x_coord), max(0, y_coord)
                x2, y2 = min(frame_width -1 , x_coord + w_val), min(frame_height -1, y_coord + h_val)
                if x2 > x1 and y2 > y1 :
                    detected_objects_info.append({'label': class_name, 'confidence': confidences[i], 
                                                  'bbox': (x1, y1, x2-x1, y2-y1)}) 
    except Exception as e:
        print(f"Error during YOLO detection: {e}")
    return detected_objects_info

# --- Main Loop ---
def main_loop():
    global USE_GUI
    speak(f"Initializing {AI_NAME_LONG} systems.", False)
    if net is None or not CLASS_NAMES: speak("Critical error: Detection model/class names not loaded.", True)
    if engine is None: speak("Warning: pyttsx3 voice synthesis not available.", True)

    cap = initialize_camera()
    if not cap: return
    speak("Systems online. I am now observing.", True)

    active_trackers = [] # List of dicts: {'id': int, 'tracker': cv2.Tracker, 'label': str, 'bbox': tuple, 'frames_since_seen': int, 'active': bool}
    next_tracker_id = 0
    
    frame_count = 0
    DETECTION_INTERVAL_FRAMES = 15 # Run YOLO detection less frequently for multi-tracking
    
    last_general_speech_time = time.time()
    general_speech_interval = 12
    announced_labels_in_tracking = set() # To announce new *types* of tracked objects once

    gui_initialized_successfully = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                speak("Error: Can't receive frame. Exiting...", True); break
            
            frame_count += 1
            current_frame_for_drawing = frame.copy()

            # 1. Update existing trackers
            lost_track_of_any = False
            for track_info in active_trackers:
                if track_info['active']:
                    success, new_bbox = track_info['tracker'].update(frame)
                    if success:
                        track_info['bbox'] = tuple(map(int, new_bbox))
                        # track_info['frames_since_seen'] = 0 # Reset if tracker update is considered "seen"
                                                            # Or only reset on YOLO re-detection match
                        if USE_GUI:
                            p1 = (track_info['bbox'][0], track_info['bbox'][1])
                            p2 = (track_info['bbox'][0] + track_info['bbox'][2], track_info['bbox'][1] + track_info['bbox'][3])
                            cv2.rectangle(current_frame_for_drawing, p1, p2, (255, 0, 0), 2, 1) # Blue
                            cv2.putText(current_frame_for_drawing, f"T{track_info['id']}:{track_info['label']}",
                                        (p1[0], p1[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
                    else:
                        track_info['active'] = False
                        lost_track_of_any = True
                        # Don't speak here yet, wait for cleanup phase or YOLO re-detection

            # 2. Periodic YOLO Detection and Track Management
            yolo_detections_this_cycle = []
            if frame_count % DETECTION_INTERVAL_FRAMES == 0:
                yolo_detections_this_cycle = detect_objects_in_frame(frame)
                
                # Increment frames_since_seen for tracks not refreshed by current YOLO detections
                for track_info in active_trackers:
                    if track_info['active']: # Only for active ones
                        track_info['frames_since_seen'] += 1

                for det in yolo_detections_this_cycle:
                    det_bbox = det['bbox']
                    det_label = det['label']
                    is_new_object = True

                    # Check against existing active tracks
                    for track_info in active_trackers:
                        if track_info['active'] and track_info['label'] == det_label:
                            iou = calculate_iou(det_bbox, track_info['bbox'])
                            if iou > IOU_THRESHOLD_FOR_NEW_TRACK: # Consider it the same object
                                is_new_object = False
                                track_info['bbox'] = det_bbox # Update bbox with more accurate YOLO detection
                                track_info['frames_since_seen'] = 0 # Refreshed by YOLO
                                # Re-initialize tracker for better accuracy if desired (can be costly)
                                # track_info['tracker'] = create_tracker_instance(TRACKER_TYPE)
                                # if track_info['tracker']: track_info['tracker'].init(frame, det_bbox)
                                break 
                    
                    if is_new_object and len(active_trackers) < MAX_TRACKERS:
                        new_tracker_obj = create_tracker_instance(TRACKER_TYPE)
                        if new_tracker_obj:
                            init_success = new_tracker_obj.init(frame, det_bbox)
                            if init_success:
                                active_trackers.append({
                                    'id': next_tracker_id, 'tracker': new_tracker_obj, 
                                    'label': det_label, 'bbox': det_bbox,
                                    'frames_since_seen': 0, 'active': True
                                })
                                if det_label not in announced_labels_in_tracking:
                                    speak(f"I've started tracking a {det_label}.", True)
                                    announced_labels_in_tracking.add(det_label)
                                next_tracker_id += 1
                            # else: speak(f"Failed to init tracker for new {det_label}", True) # Too verbose

                # Cleanup inactive/stale trackers
                updated_active_trackers = []
                for track_info in active_trackers:
                    if track_info['active'] and track_info['frames_since_seen'] <= MAX_FRAMES_SINCE_SEEN_THRESHOLD:
                        updated_active_trackers.append(track_info)
                    else:
                        if not track_info['active']: # Lost by tracker.update()
                             speak(f"I've lost track of {track_info['label']} (ID {track_info['id']}).", True)
                        elif track_info['frames_since_seen'] > MAX_FRAMES_SINCE_SEEN_THRESHOLD: # Lost by staleness
                             speak(f"{track_info['label']} (ID {track_info['id']}) seems to be gone.", True)
                        
                        if track_info['label'] in announced_labels_in_tracking:
                            # Check if any other instance of this label is still being tracked
                            if not any(t['label'] == track_info['label'] and t['active'] for t in active_trackers if t['id'] != track_info['id']):
                                announced_labels_in_tracking.discard(track_info['label'])
                active_trackers = updated_active_trackers


            # 3. General Speech for non-tracked but detected objects (from current YOLO cycle)
            current_time = time.time()
            if yolo_detections_this_cycle and (current_time - last_general_speech_time > general_speech_interval):
                objects_to_mention_general = []
                current_tracked_labels_with_ids = {f"{t['label']}_T{t['id']}" for t in active_trackers if t['active']}

                for det in yolo_detections_this_cycle:
                    # Avoid general announcement if a similar object is actively tracked (even if different ID)
                    is_actively_tracked_type = False
                    for track_info in active_trackers:
                        if track_info['active'] and track_info['label'] == det['label']:
                            iou_with_tracked = calculate_iou(det['bbox'], track_info['bbox'])
                            if iou_with_tracked > 0.5: # If a YOLO box strongly overlaps an active track of same type
                                is_actively_tracked_type = True
                                break
                    if not is_actively_tracked_type:
                         objects_to_mention_general.append(det['label'])
                
                # Make general announcements less repetitive
                unique_general_mentions = list(set(objects_to_mention_general))
                if unique_general_mentions:
                    if len(unique_general_mentions) == 1:
                        speak(f"I also see a {unique_general_mentions[0]} in the scene.")
                    else:
                        items_speech = "a " + ", a ".join(unique_general_mentions[:-1]) + f", and a {unique_general_mentions[-1]}"
                        speak(f"Also in the scene: {items_speech}.")
                last_general_speech_time = current_time


            # 4. Display
            if USE_GUI:
                try:
                    # Draw YOLO boxes from the latest detection cycle (green)
                    if frame_count % DETECTION_INTERVAL_FRAMES == 0:
                        for det in yolo_detections_this_cycle:
                            is_being_actively_tracked = False
                            for track_info in active_trackers:
                                if track_info['active'] and track_info['label'] == det['label'] and \
                                   calculate_iou(det['bbox'], track_info['bbox']) > 0.5 : # Check if this yolo box corresponds to an active track
                                    is_being_actively_tracked = True
                                    break
                            if not is_being_actively_tracked: # Only draw YOLO box if not actively tracked (blue)
                                x, y, w, h = det['bbox']
                                cv2.rectangle(current_frame_for_drawing, (x,y), (x+w, y+h), (0,255,0), 2)
                                cv2.putText(current_frame_for_drawing, f"{det['label']}: {det['confidence']:.2f}",
                                            (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

                    cv2.imshow(f'{AI_NAME_SHORT} Vision (Multi-Tracking) - Q to Quit', current_frame_for_drawing)
                    gui_initialized_successfully = True
                except cv2.error as e:
                    if "The function is not implemented" in str(e):
                        if not gui_initialized_successfully: print_opencv_gui_warning(); speak("Warning: Display disabled.",True)
                        USE_GUI = False 
                    else: print(f"imshow error: {e}"); speak("Display error.",True); USE_GUI = False 
            
            key_press = cv2.waitKey(1) & 0xFF
            if key_press == ord('q'): speak("Deactivating.", True); break

    except KeyboardInterrupt: speak("Deactivation requested.", True)
    finally:
        if cap: cap.release()
        if gui_initialized_successfully:
            try: cv2.destroyAllWindows()
            except cv2.error: pass # Ignore error if window already closed
        speak(f"{AI_NAME_LONG} systems offline.", False)

def print_opencv_gui_warning():
    print("\n*** OpenCV GUI Error: Cannot display camera window. Ensure 'opencv-contrib-python' is installed and GUI libs are available. Disabling GUI. ***\n")

if __name__ == '__main__':
    if not all([os.path.exists(MODEL_CONFIG), os.path.exists(MODEL_WEIGHTS), os.path.exists(CLASS_LABELS_FILE)]):
        print("\n--- CRITICAL ERROR: Model files missing. Please check 'dnn_model' directory. Exiting. ---")
    else:
        speak(f"Hello! I am {AI_NAME_LONG}, but you can call me {AI_NAME_SHORT}.", False)
        main_loop()
