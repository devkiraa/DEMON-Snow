# Import necessary libraries
import cv2
import pyttsx3
import time
import os
import numpy as np
import threading
import queue # For thread-safe communication

# --- Configuration ---
AI_NAME_LONG = "Deamon Snow"
AI_NAME_SHORT = "Snow"
USE_GUI = True 

# --- Field Operation Configuration ---
STEALTH_MODE = False 
LOW_LIGHT_MODE = False 
SOUND_DETECTION_MODE = False 
SYSTEM_STATUS_REPORTING = True 

HIGH_PRIORITY_CLASSES = ["person", "car", "truck", "bus", "motorcycle", "bicycle"] 
FOCUS_ZONE_ENABLED = True 
FOCUS_ZONE_RATIOS = (0.25, 0.25, 0.75, 0.75) 

# --- pyttsx3 Configuration ---
# Engine will be initialized in the speech worker thread
engine = None # Initialize as None globally

# --- DNN Model Configuration (YOLOv3-tiny) ---
MODEL_DIR = "dnn_model"
MODEL_CONFIG = os.path.join(MODEL_DIR, "yolov3-tiny.cfg")
MODEL_WEIGHTS = os.path.join(MODEL_DIR, "yolov3-tiny.weights")
CLASS_LABELS_FILE = os.path.join(MODEL_DIR, "coco.names")

CONFIDENCE_THRESHOLD = 0.55 
NMS_THRESHOLD = 0.4        
DNN_INPUT_SIZE = (416, 416) 

# --- Tracker Configuration ---
TRACKER_TYPE = "CSRT" 
IOU_THRESHOLD_FOR_NEW_TRACK = 0.3 
MAX_TRACKERS = 7 
MAX_FRAMES_SINCE_SEEN_THRESHOLD = 3 

# --- Threading Queues ---
frame_queue = queue.Queue(maxsize=2) 
detection_results_queue = queue.Queue(maxsize=2)
speech_queue = queue.Queue(maxsize=10) # Queue for text to be spoken

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
def speak(text, use_short_name=True, is_priority=False):
    """Prints text to console and adds it to the speech queue."""
    if STEALTH_MODE and not is_priority: return

    ai_speaker_name = AI_NAME_SHORT if use_short_name else AI_NAME_LONG
    prefix = "ALERT: " if is_priority else ""
    full_text_console = f"{ai_speaker_name}: {prefix}{text}"
    print(full_text_console) # Immediate console output
    
    # Add text to speech queue for background processing
    try:
        speech_queue.put_nowait({'text': f"{prefix}{text}", 'is_priority': is_priority})
    except queue.Full:
        print(f"{AI_NAME_SHORT}: Speech queue full, message '{text}' dropped.")


def initialize_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        speak("CRITICAL: Could not open camera.", False, is_priority=True)
        return None
    speak("Camera initialized.", False) # This will now be non-blocking
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
    except Exception: return []

def create_tracker_instance(tracker_type_str):
    tracker = None
    try:
        if tracker_type_str == "CSRT":
            if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
                tracker = cv2.legacy.TrackerCSRT_create()
            elif hasattr(cv2, 'TrackerCSRT_create'): 
                tracker = cv2.TrackerCSRT_create()
        if tracker is None: print(f"Warning: Could not create tracker: {tracker_type_str}.")
    except Exception as e: print(f"Error creating tracker '{tracker_type_str}': {e}")
    return tracker

def calculate_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[0] + boxA[2], boxB[0] + boxB[2]), min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0
    boxAArea, boxBArea = boxA[2] * boxA[3], boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea)

def enhance_frame_for_low_light(frame_to_enhance):
    if frame_to_enhance is None: return None
    try:
        hsv = cv2.cvtColor(frame_to_enhance, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v_eq = cv2.equalizeHist(v)
        enhanced_frame = cv2.cvtColor(cv2.merge([h, s, v_eq]), cv2.COLOR_HSV2BGR)
        return enhanced_frame
    except cv2.error: return frame_to_enhance

def is_in_focus_zone(bbox, frame_w, frame_h, zone_ratios):
    if not FOCUS_ZONE_ENABLED or bbox is None: return False
    fx1, fy1 = int(zone_ratios[0] * frame_w), int(zone_ratios[1] * frame_h)
    fx2, fy2 = int(zone_ratios[2] * frame_w), int(zone_ratios[3] * frame_h)
    obj_center_x, obj_center_y = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
    return fx1 <= obj_center_x <= fx2 and fy1 <= obj_center_y <= fy2

def get_object_direction(bbox, frame_w):
    if bbox is None: return ""
    center_x = bbox[0] + bbox[2] / 2
    third_width = frame_w / 3
    if center_x < third_width: return "left"
    elif center_x > 2 * third_width: return "right"
    else: return "center"

# --- Speech Worker Thread Function ---
def speech_worker(q, stop_event):
    """Worker thread for text-to-speech using pyttsx3."""
    global engine # Use the global engine variable
    print("Speech worker thread started.")
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        if voices and len(voices) > 1:
            try: engine.setProperty('voice', voices[1].id) 
            except Exception: 
                if voices: engine.setProperty('voice', voices[0].id) 
        elif voices: engine.setProperty('voice', voices[0].id)
        engine.setProperty('rate', 175)
        engine.setProperty('volume', 1.0)
        print("pyttsx3 engine initialized in speech thread.")
    except Exception as e:
        print(f"Error initializing pyttsx3 engine in speech thread: {e}")
        engine = None # Ensure engine is None if init fails

    while not stop_event.is_set():
        try:
            speech_data = q.get(timeout=0.1) # Wait for speech text
            if speech_data is None: continue # Should not happen with current logic but good practice

            text_to_say = speech_data['text']
            # is_priority_speech = speech_data['is_priority'] # Can be used for future logic if needed

            if engine:
                try:
                    engine.say(text_to_say)
                    engine.runAndWait() # This is blocking for this thread, but not the main thread
                except Exception as e:
                    print(f"Error during pyttsx3 speech synthesis: {e}")
            else:
                # This console log might be redundant if speak() already prints
                # print(f"Speech Engine (in thread) not ready for: {text_to_say}") 
                pass
            q.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in speech worker: {e}")
            time.sleep(0.1)
    print("Speech worker thread stopped.")


# --- Detection Thread Function ---
def detection_worker(frame_q, results_q, stop_event):
    print("Detection worker thread started.")
    local_net = net 
    
    while not stop_event.is_set():
        try:
            frame_data = frame_q.get(timeout=0.1) 
            if frame_data is None: continue 
            
            frame_to_detect, frame_w_local, frame_h_local = frame_data
            detected_objects_info = []
            if local_net and CLASS_NAMES and frame_to_detect is not None:
                try:
                    blob = cv2.dnn.blobFromImage(frame_to_detect, 1/255.0, DNN_INPUT_SIZE, swapRB=True, crop=False)
                    local_net.setInput(blob)
                    output_layers_local = get_output_layers(local_net) 
                    if output_layers_local:
                        layer_outputs = local_net.forward(output_layers_local)
                        boxes, confidences, class_ids = [], [], []
                        for output in layer_outputs:
                            for detection_data in output: 
                                scores = detection_data[5:] 
                                class_id = np.argmax(scores)
                                confidence = scores[class_id]
                                if confidence > CONFIDENCE_THRESHOLD:
                                    center_x = int(detection_data[0] * frame_w_local)
                                    center_y = int(detection_data[1] * frame_h_local)
                                    w = int(detection_data[2] * frame_w_local)
                                    h = int(detection_data[3] * frame_h_local)
                                    x = int(center_x - w / 2)
                                    y = int(center_y - h / 2)
                                    boxes.append([x, y, w, h])
                                    confidences.append(float(confidence))
                                    class_ids.append(class_id)

                        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
                        if len(indices) > 0:
                            processed_indices = indices.flatten() if isinstance(indices, np.ndarray) and indices.ndim == 2 else indices
                            for i in processed_indices:
                                box = boxes[i]
                                x_coord, y_coord, w_val, h_val = box[0], box[1], box[2], box[3]
                                class_name = CLASS_NAMES[class_ids[i]] if class_ids[i] < len(CLASS_NAMES) else "Unknown"
                                x1, y1 = max(0, x_coord), max(0, y_coord)
                                x2, y2 = min(frame_w_local -1 , x_coord + w_val), min(frame_h_local -1, y_coord + h_val)
                                if x2 > x1 and y2 > y1 :
                                    bbox_data = (x1, y1, x2-x1, y2-y1)
                                    in_focus = is_in_focus_zone(bbox_data, frame_w_local, frame_h_local, FOCUS_ZONE_RATIOS)
                                    is_priority_class = class_name in HIGH_PRIORITY_CLASSES
                                    detected_objects_info.append({'label': class_name, 'confidence': confidences[i], 
                                                                  'bbox': bbox_data, 'in_focus': in_focus, 
                                                                  'is_priority': is_priority_class})
                except Exception as e:
                    print(f"Detection thread error: {e}")
            try:
                results_q.put(detected_objects_info, block=False) 
            except queue.Full: pass
            frame_q.task_done()
        except queue.Empty: continue 
        except Exception as e:
            print(f"Outer detection worker error: {e}")
            time.sleep(0.1) 
    print("Detection worker thread stopped.")


# --- Main Loop ---
def main_loop():
    global USE_GUI, STEALTH_MODE, LOW_LIGHT_MODE, SOUND_DETECTION_MODE, SYSTEM_STATUS_REPORTING
    
    # Start speech worker thread first, so speak() can queue messages during init
    stop_speech_event = threading.Event()
    speech_thread = threading.Thread(target=speech_worker, args=(speech_queue, stop_speech_event))
    speech_thread.daemon = True
    speech_thread.start()
    time.sleep(0.5) # Give speech thread a moment to initialize its engine

    speak(f"Initializing {AI_NAME_LONG} systems.", False) # This will now queue
    if net is None or not CLASS_NAMES: speak("CRITICAL: Detection model/class names not loaded.", True, is_priority=True)
    # Engine status will be known by the speech thread itself

    cap = initialize_camera()
    if not cap: 
        stop_speech_event.set() # Stop speech thread if camera fails
        if speech_thread.is_alive(): speech_thread.join(timeout=1)
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fz_x1, fz_y1, fz_x2, fz_y2 = 0,0,0,0
    if FOCUS_ZONE_ENABLED:
        fz_x1, fz_y1 = int(FOCUS_ZONE_RATIOS[0] * frame_width), int(FOCUS_ZONE_RATIOS[1] * frame_height)
        fz_x2, fz_y2 = int(FOCUS_ZONE_RATIOS[2] * frame_width), int(FOCUS_ZONE_RATIOS[3] * frame_height)

    speak("Systems online. Awaiting visual input.", True)

    active_trackers = [] 
    next_tracker_id = 0
    
    stop_detection_event = threading.Event()
    detection_thread = threading.Thread(target=detection_worker, args=(frame_queue, detection_results_queue, stop_detection_event))
    detection_thread.daemon = True 
    detection_thread.start()

    yolo_detections_this_cycle = [] 

    last_general_speech_time = time.time()
    general_speech_interval = 15 
    announced_labels_in_tracking = set() 
    last_status_report_time = time.time()
    status_report_interval = 30 

    gui_initialized_successfully = False
    simulated_sound_event_triggered = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                speak("CRITICAL: Frame acquisition failed. Terminating.", True, is_priority=True); break
            
            try:
                frame_queue.put_nowait((frame.copy(), frame_width, frame_height)) 
            except queue.Full: pass

            try:
                yolo_detections_this_cycle = detection_results_queue.get_nowait()
            except queue.Empty: pass

            processed_frame = frame.copy() 
            if LOW_LIGHT_MODE:
                processed_frame = enhance_frame_for_low_light(processed_frame)
            current_frame_for_drawing = processed_frame.copy()

            if simulated_sound_event_triggered:
                speak("CRITICAL SOUND EVENT DETECTED!", True, is_priority=True)
                simulated_sound_event_triggered = False 

            for track_info in active_trackers:
                if track_info['active']:
                    success, new_bbox = track_info['tracker'].update(processed_frame)
                    if success:
                        track_info['bbox'] = tuple(map(int, new_bbox))
                        track_info['in_focus'] = is_in_focus_zone(track_info['bbox'], frame_width, frame_height, FOCUS_ZONE_RATIOS)
                    else:
                        track_info['active'] = False 

            # Track Management
            for track_info in active_trackers: 
                if track_info['active']: track_info['frames_since_seen'] += 1
            
            for det in yolo_detections_this_cycle: 
                det_bbox, det_label = det['bbox'], det['label']
                det_in_focus, det_is_priority = det['in_focus'], det['is_priority']
                is_new_object = True

                for track_info in active_trackers:
                    if track_info['active'] and track_info['label'] == det_label:
                        if calculate_iou(det_bbox, track_info['bbox']) > IOU_THRESHOLD_FOR_NEW_TRACK: 
                            is_new_object = False
                            track_info['bbox'], track_info['frames_since_seen'] = det_bbox, 0
                            track_info['in_focus'], track_info['is_priority'] = det_in_focus, det_is_priority
                            break 
                
                if is_new_object and len(active_trackers) < MAX_TRACKERS:
                    new_tracker_obj = create_tracker_instance(TRACKER_TYPE)
                    if new_tracker_obj and new_tracker_obj.init(processed_frame, det_bbox):
                        direction = get_object_direction(det_bbox, frame_width)
                        active_trackers.append({
                            'id': next_tracker_id, 'tracker': new_tracker_obj, 'label': det_label, 
                            'bbox': det_bbox, 'frames_since_seen': 0, 'active': True, 
                            'is_priority': det_is_priority, 'in_focus': det_in_focus, 'direction': direction
                        })
                        if det_label not in announced_labels_in_tracking or det_is_priority:
                            focus_text = " in focus zone" if det_in_focus else ""
                            dir_text = f", {direction}" if direction else ""
                            speak(f"Tracking new {det_label}{focus_text}{dir_text}.", True, is_priority=det_is_priority)
                            if not det_is_priority: announced_labels_in_tracking.add(det_label)
                        next_tracker_id += 1

            # Cleanup
            updated_active_trackers = []
            for track_info in active_trackers:
                if track_info['active'] and track_info['frames_since_seen'] <= MAX_FRAMES_SINCE_SEEN_THRESHOLD:
                    updated_active_trackers.append(track_info)
                else:
                    lost_reason = "lost by tracker" if not track_info['active'] else "stale"
                    speak(f"{track_info['label']} (ID {track_info['id']}) {lost_reason}.", True, is_priority=track_info.get('is_priority', False))
                    if track_info['label'] in announced_labels_in_tracking and not track_info.get('is_priority', False):
                        if not any(t['label'] == track_info['label'] and t['active'] and not t.get('is_priority', False) for t in active_trackers if t['id'] != track_info['id']):
                            announced_labels_in_tracking.discard(track_info['label'])
            active_trackers = updated_active_trackers
            
            current_time = time.time()
            if yolo_detections_this_cycle and (current_time - last_general_speech_time > general_speech_interval):
                # ... (General speech logic can be further refined if needed)
                last_general_speech_time = current_time

            if SYSTEM_STATUS_REPORTING and (current_time - last_status_report_time > status_report_interval):
                tracked_count = sum(1 for t in active_trackers if t['active'])
                status_msg = f"System status: {tracked_count} objects tracked. "
                if STEALTH_MODE: status_msg += "Stealth ON. "
                if LOW_LIGHT_MODE: status_msg += "Low Light ON. "
                if SOUND_DETECTION_MODE: status_msg += "Audio Monitoring ON. "
                speak(status_msg, True, is_priority=False) 
                last_status_report_time = current_time

            if USE_GUI:
                try:
                    for track_info in active_trackers:
                        if track_info['active'] and track_info['bbox']: 
                            p1 = (track_info['bbox'][0], track_info['bbox'][1])
                            p2 = (track_info['bbox'][0] + track_info['bbox'][2], track_info['bbox'][1] + track_info['bbox'][3])
                            color = (255, 0, 0) 
                            if track_info.get('is_priority', False):
                                color = (0, 0, 255) 
                                if track_info['in_focus']: color = (0, 165, 255) 
                            cv2.rectangle(current_frame_for_drawing, p1, p2, color, 2, 1) 
                            label_text = f"T{track_info['id']}:{track_info['label']}"
                            if track_info['in_focus']: label_text += " (FZ)"
                            if track_info.get('direction'): label_text += f" {track_info['direction'][0].upper()}" 
                            cv2.putText(current_frame_for_drawing, label_text, (p1[0], p1[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
                    for det in yolo_detections_this_cycle:
                        is_being_actively_tracked = any(t['active'] and t['label'] == det['label'] and calculate_iou(det['bbox'], t['bbox']) > 0.5 for t in active_trackers)
                        if not is_being_actively_tracked:
                            x, y, w, h = det['bbox']
                            color = (0, 255, 0) 
                            if det['is_priority']: color = (0, 128, 0) 
                            if det['in_focus'] and det['is_priority']: color = (0,200,0)
                            cv2.rectangle(current_frame_for_drawing, (x,y), (x+w, y+h), color, 2)
                            label_text = f"{det['label']}: {det['confidence']:.2f}"
                            if det['in_focus']: label_text += " (FZ)"
                            cv2.putText(current_frame_for_drawing, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                    if FOCUS_ZONE_ENABLED:
                        cv2.rectangle(current_frame_for_drawing, (fz_x1, fz_y1), (fz_x2, fz_y2), (0, 255, 255), 1) 
                        cv2.putText(current_frame_for_drawing, "Focus Zone", (fz_x1, fz_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255),1)
                    
                    title_status = f"S:{'ON' if STEALTH_MODE else 'OFF'}|L:{'ON' if LOW_LIGHT_MODE else 'OFF'}|A:{'ON' if SOUND_DETECTION_MODE else 'OFF'}|R:{'ON' if SYSTEM_STATUS_REPORTING else 'OFF'}"
                    cv2.imshow(f'{AI_NAME_SHORT} Field Ops ({title_status}) - Q:Quit', current_frame_for_drawing)
                    gui_initialized_successfully = True
                except cv2.error as e:
                    if "The function is not implemented" in str(e):
                        if not gui_initialized_successfully: print_opencv_gui_warning(); speak("Warning: Display disabled.",True, is_priority=True)
                        USE_GUI = False 
                    else: print(f"imshow error: {e}"); speak("Display error.",True, is_priority=True); USE_GUI = False 
            
            key_press = cv2.waitKey(1) & 0xFF 
            if key_press == ord('q'): speak("Deactivating system.", True, is_priority=True); break
            elif key_press == ord('s'):
                STEALTH_MODE = not STEALTH_MODE
                speak(f"Stealth Mode {'ENGAGED' if STEALTH_MODE else 'DISENGAGED'}.", True, is_priority=True)
            elif key_press == ord('l'):
                LOW_LIGHT_MODE = not LOW_LIGHT_MODE
                speak(f"Low Light Enhancement {'ACTIVATED' if LOW_LIGHT_MODE else 'DEACTIVATED'}.", True, is_priority=True)
            elif key_press == ord('a'): 
                SOUND_DETECTION_MODE = not SOUND_DETECTION_MODE
                speak(f"Simulated Sound Detection {'ENABLED' if SOUND_DETECTION_MODE else 'DISABLED'}.", True, is_priority=True)
            elif key_press == ord('x'): 
                if SOUND_DETECTION_MODE:
                    simulated_sound_event_triggered = True 
                else:
                    speak("Sound detection mode is off. Cannot simulate event.", True)
            elif key_press == ord('r'): 
                SYSTEM_STATUS_REPORTING = not SYSTEM_STATUS_REPORTING
                speak(f"System Status Reporting {'ENABLED' if SYSTEM_STATUS_REPORTING else 'DISABLED'}.", True, is_priority=True)

    except KeyboardInterrupt: speak("Deactivation by user interrupt.", True, is_priority=True)
    finally:
        print("Shutting down threads...")
        stop_detection_event.set() 
        stop_speech_event.set() # Signal speech thread to stop

        if detection_thread.is_alive():
            try: frame_queue.put_nowait(None) 
            except queue.Full: pass
            detection_thread.join(timeout=1) 
        
        if speech_thread.is_alive():
            try: speech_queue.put_nowait({'text': "Finalizing shutdown.", 'is_priority': False}) # Send one last item to unblock .get()
            except queue.Full: pass
            speech_thread.join(timeout=2) # Wait for speech thread
        
        if cap: cap.release()
        if gui_initialized_successfully:
            try: cv2.destroyAllWindows()
            except cv2.error: pass 
        # Use print for the very final message as speech thread might be fully stopped
        print(f"{AI_NAME_LONG}: Systems offline.")


def print_opencv_gui_warning():
    print("\n*** OpenCV GUI Error: Cannot display camera window. Ensure 'opencv-contrib-python' is installed and GUI libs are available. Disabling GUI. ***\n")

if __name__ == '__main__':
    if not all([os.path.exists(MODEL_CONFIG), os.path.exists(MODEL_WEIGHTS), os.path.exists(CLASS_LABELS_FILE)]):
        print("\n--- CRITICAL ERROR: Model files missing. Please check 'dnn_model' directory. Exiting. ---")
    else:
        # Initial greeting will be queued and spoken by the speech thread
        speak(f"Hello! I am {AI_NAME_LONG}, but you can call me {AI_NAME_SHORT}. System check nominal.", False)
        main_loop()
