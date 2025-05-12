# Import necessary libraries
import cv2
import pyttsx3
import time
import os
import numpy as np
import threading
import queue # For thread-safe communication
from ultralytics import YOLO # Import Ultralytics YOLO
import pytesseract # Import pytesseract for OCR

# --- Tesseract Configuration ---
# Action Required: Install Tesseract OCR engine separately.
# If pytesseract cannot find your Tesseract installation automatically, 
# uncomment and set the correct path below.
# Example for Windows:
# try:
#     pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# except FileNotFoundError:
#      print("Tesseract command path not found, ensure Tesseract is installed and path is correct (if needed).")
# except Exception as e:
#      print(f"Error setting Tesseract path: {e}")


# --- General Configuration ---
AI_NAME_LONG = "Deamon Snow"
AI_NAME_SHORT = "Snow"
USE_GUI = True 

# --- Field Operation Configuration ---
STEALTH_MODE = False 
LOW_LIGHT_MODE = False 
SOUND_DETECTION_MODE = False 
SYSTEM_STATUS_REPORTING = True 
OCR_ENABLED = False # Optical Character Recognition (Text Detection) - Toggle with 'T' key

HIGH_PRIORITY_CLASSES = ["person", "car", "truck", "bus", "motorcycle", "bicycle"] 
FOCUS_ZONE_ENABLED = True 
FOCUS_ZONE_RATIOS = (0.25, 0.25, 0.75, 0.75) # Focus OCR on this central area

# --- pyttsx3 Configuration ---
engine = None 

# --- DNN Model Configuration (Ultralytics YOLOv8m) ---
MODEL_DIR = "dnn_model" 
CLASS_LABELS_FILE = os.path.join(MODEL_DIR, "coco.names")
YOLO_MODEL_NAME = 'yolov11m.pt' # UPGRADED to Medium model

CONFIDENCE_THRESHOLD = 0.55 # Confidence threshold for YOLO detections
# NMS_THRESHOLD handled by ultralytics
# DNN_INPUT_SIZE handled by ultralytics

# --- Tracker Configuration ---
TRACKER_TYPE = "CSRT" 
IOU_THRESHOLD_FOR_NEW_TRACK = 0.3 
MAX_TRACKERS = 4 # Reduced slightly more for heavier model
MAX_FRAMES_SINCE_SEEN_THRESHOLD = 4 

# --- OCR Configuration ---
OCR_INTERVAL_CYCLES = 3 # Run OCR every N detection cycles
OCR_CONFIDENCE_THRESHOLD = 40 # Tesseract confidence threshold (0-100)

# --- Threading Queues ---
frame_queue = queue.Queue(maxsize=1) 
detection_results_queue = queue.Queue(maxsize=1) # Will now include OCR results
speech_queue = queue.Queue(maxsize=10) 

# Load class names
try:
    with open(CLASS_LABELS_FILE, 'rt') as f:
        CLASS_NAMES = f.read().rstrip('\n').split('\n')
except FileNotFoundError:
    print(f"Error: Class labels file '{CLASS_LABELS_FILE}' not found.")
    CLASS_NAMES = []

# Initialize the Ultralytics YOLO model
try:
    model = YOLO(YOLO_MODEL_NAME, verbose=False) 
    print(f"Ultralytics YOLO model '{YOLO_MODEL_NAME}' initialized.")
except Exception as e:
    print(f"Error initializing Ultralytics YOLO model: {e}")
    model = None

# --- Helper Functions ---
def speak(text, use_short_name=True, is_priority=False):
    if STEALTH_MODE and not is_priority: return
    ai_speaker_name = AI_NAME_SHORT if use_short_name else AI_NAME_LONG
    prefix = "ALERT: " if is_priority else ""
    full_text_console = f"{ai_speaker_name}: {prefix}{text}"
    print(full_text_console)
    try:
        speech_queue.put_nowait({'text': f"{prefix}{text}", 'is_priority': is_priority})
    except queue.Full: pass # Ignore if queue is full

def initialize_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        speak("CRITICAL: Could not open camera.", False, is_priority=True)
        return None
    speak("Camera initialized.", False)
    return cap

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
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

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
    global engine 
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
        engine = None 

    while not stop_event.is_set():
        try:
            speech_data = q.get(timeout=0.1) 
            if speech_data is None: continue 
            text_to_say = speech_data['text']
            if engine:
                try:
                    engine.say(text_to_say)
                    engine.runAndWait() 
                except Exception as e:
                    print(f"Error during pyttsx3 speech synthesis: {e}")
            q.task_done()
        except queue.Empty: continue
        except Exception as e: print(f"Error in speech worker: {e}"); time.sleep(0.1)
    print("Speech worker thread stopped.")

# --- Detection & OCR Thread Function ---
def detection_worker(frame_q, results_q, stop_event):
    print("Detection worker thread started.")
    local_model = model 
    detection_cycle_count = 0
    
    while not stop_event.is_set():
        try:
            frame_data = frame_q.get(timeout=0.2) 
            if frame_data is None: continue 
            
            frame_to_detect, frame_w_local, frame_h_local = frame_data
            detected_objects_info = []
            ocr_results_text = None # Initialize OCR results for this cycle

            # --- Object Detection ---
            if local_model and CLASS_NAMES and frame_to_detect is not None:
                try:
                    results = local_model.predict(frame_to_detect, conf=CONFIDENCE_THRESHOLD, verbose=False) 
                    for box in results[0].boxes:
                        class_id = int(box.cls.item()) 
                        confidence = box.conf.item() 
                        if class_id < len(CLASS_NAMES): 
                            class_name = CLASS_NAMES[class_id]
                            xyxy = box.xyxy.cpu().numpy().flatten()
                            x1, y1, x2, y2 = map(int, xyxy)
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(frame_w_local - 1, x2), min(frame_h_local - 1, y2)
                            if x2 > x1 and y2 > y1: 
                                bbox_data = (x1, y1, x2 - x1, y2 - y1) 
                                in_focus = is_in_focus_zone(bbox_data, frame_w_local, frame_h_local, FOCUS_ZONE_RATIOS)
                                is_priority_class = class_name in HIGH_PRIORITY_CLASSES
                                detected_objects_info.append({'label': class_name, 'confidence': confidence, 
                                                              'bbox': bbox_data, 'in_focus': in_focus, 
                                                              'is_priority': is_priority_class})
                except Exception as e:
                    print(f"Detection thread error during YOLO processing: {e}") 

            # --- OCR Processing (Periodic & Enabled) ---
            detection_cycle_count += 1
            if OCR_ENABLED and (detection_cycle_count % OCR_INTERVAL_CYCLES == 0):
                try:
                    # Extract Focus Zone for OCR
                    if FOCUS_ZONE_ENABLED and frame_to_detect is not None:
                        fz_x1_ocr = int(FOCUS_ZONE_RATIOS[0] * frame_w_local)
                        fz_y1_ocr = int(FOCUS_ZONE_RATIOS[1] * frame_h_local)
                        fz_x2_ocr = int(FOCUS_ZONE_RATIOS[2] * frame_w_local)
                        fz_y2_ocr = int(FOCUS_ZONE_RATIOS[3] * frame_h_local)
                        
                        focus_roi = frame_to_detect[fz_y1_ocr:fz_y2_ocr, fz_x1_ocr:fz_x2_ocr]

                        if focus_roi.size > 0: # Check if ROI is valid
                            # Convert ROI to grayscale for potentially better OCR
                            gray_roi = cv2.cvtColor(focus_roi, cv2.COLOR_BGR2GRAY)
                            # Optional: Apply thresholding or other preprocessing
                            # _, thresh_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                            
                            # Perform OCR using pytesseract
                            # Use --psm 6 for assuming a single uniform block of text
                            custom_config = r'--oem 3 --psm 6' 
                            ocr_data = pytesseract.image_to_data(gray_roi, config=custom_config, output_type=pytesseract.Output.DICT)
                            
                            # Extract text with confidence > threshold
                            extracted_texts = []
                            for i in range(len(ocr_data['text'])):
                                if int(ocr_data['conf'][i]) > OCR_CONFIDENCE_THRESHOLD:
                                    text = ocr_data['text'][i].strip()
                                    if text: # Ignore empty strings
                                        extracted_texts.append(text)
                            
                            if extracted_texts:
                                ocr_results_text = " ".join(extracted_texts)
                                # print(f"OCR Detected Text: {ocr_results_text}") # Print in thread for debug
                        # else: print("OCR Focus Zone ROI is empty.") # Debug
                    # else: print("OCR skipped: Focus zone disabled or invalid frame.") # Debug

                except pytesseract.TesseractNotFoundError:
                     print("OCR Error: Tesseract executable not found or not in PATH.")
                     print("Please install Tesseract and/or set pytesseract.pytesseract.tesseract_cmd")
                     # Disable OCR to prevent repeated errors in this session
                     # global OCR_ENABLED # This won't work directly, need a different mechanism if you want to disable it permanently from thread
                except Exception as e:
                    print(f"OCR thread error during processing: {e}")
            
            # --- Put results (detections and OCR) in queue ---
            try:
                results_q.put({'detections': detected_objects_info, 'ocr_text': ocr_results_text}, block=False) 
            except queue.Full: pass 
            frame_q.task_done()
        except queue.Empty: continue 
        except Exception as e:
            print(f"Outer detection worker error: {e}")
            time.sleep(0.1) 
    print("Detection worker thread stopped.")

# --- Main Loop ---
def main_loop():
    global USE_GUI, STEALTH_MODE, LOW_LIGHT_MODE, SOUND_DETECTION_MODE, SYSTEM_STATUS_REPORTING, OCR_ENABLED
    
    stop_speech_event = threading.Event()
    speech_thread = threading.Thread(target=speech_worker, args=(speech_queue, stop_speech_event))
    speech_thread.daemon = True
    speech_thread.start()
    time.sleep(0.5) 

    speak(f"Initializing {AI_NAME_LONG} systems.", False)
    if model is None or not CLASS_NAMES: speak("CRITICAL: Detection model/class names not loaded.", True, is_priority=True)

    cap = initialize_camera()
    if not cap: 
        stop_speech_event.set()
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

    latest_results = {'detections': [], 'ocr_text': None}

    last_general_speech_time = time.time()
    general_speech_interval = 20 
    announced_labels_in_tracking = set() 
    last_status_report_time = time.time()
    status_report_interval = 30 
    last_ocr_report_time = time.time()
    ocr_report_interval = 10 # How often to report detected text (if any)

    gui_initialized_successfully = False
    simulated_sound_event_triggered = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                speak("CRITICAL: Frame acquisition failed. Terminating.", True, is_priority=True); break
            
            try:
                if frame_queue.full(): frame_queue.get_nowait() 
                frame_queue.put_nowait((frame.copy(), frame_width, frame_height)) 
            except queue.Full: pass 
            except queue.Empty: pass 

            try:
                while not detection_results_queue.empty(): # Get the latest result
                    latest_results = detection_results_queue.get_nowait()
            except queue.Empty: pass 
            
            yolo_detections_this_cycle = latest_results['detections']
            ocr_text_this_cycle = latest_results['ocr_text']

            processed_frame = frame.copy() 
            if LOW_LIGHT_MODE:
                processed_frame = enhance_frame_for_low_light(processed_frame)
            current_frame_for_drawing = processed_frame.copy()

            if simulated_sound_event_triggered:
                speak("CRITICAL SOUND EVENT DETECTED!", True, is_priority=True)
                simulated_sound_event_triggered = False 

            # --- Tracker Update ---
            for track_info in active_trackers:
                if track_info['active']:
                    success, new_bbox = track_info['tracker'].update(processed_frame)
                    if success:
                        track_info['bbox'] = tuple(map(int, new_bbox))
                        track_info['in_focus'] = is_in_focus_zone(track_info['bbox'], frame_width, frame_height, FOCUS_ZONE_RATIOS)
                    else:
                        track_info['active'] = False 

            # --- Track Management ---
            for track_info in active_trackers: 
                if track_info['active']: track_info['frames_since_seen'] += 1
            
            current_tracked_object_ids = {t['id'] for t in active_trackers if t['active']}

            for det in yolo_detections_this_cycle: 
                det_bbox, det_label = det['bbox'], det['label']
                det_in_focus, det_is_priority = det['in_focus'], det['is_priority']
                is_new_object = True
                matched_track_id = -1

                best_iou = IOU_THRESHOLD_FOR_NEW_TRACK 
                for track_info in active_trackers:
                    if track_info['active'] and track_info['label'] == det_label:
                        iou = calculate_iou(det_bbox, track_info['bbox'])
                        if iou > best_iou: 
                            best_iou = iou
                            is_new_object = False
                            matched_track_id = track_info['id']
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
                        if det_is_priority or (not STEALTH_MODE and det_label not in announced_labels_in_tracking) :
                            focus_text = " in focus zone" if det_in_focus else ""
                            dir_text = f", {direction}" if direction else ""
                            speak(f"Tracking new {det_label}{focus_text}{dir_text}.", True, is_priority=det_is_priority)
                            if not det_is_priority: announced_labels_in_tracking.add(det_label)
                        next_tracker_id += 1
            
            # --- Cleanup ---
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
            
            # --- Speech: General & OCR ---
            current_time = time.time()
            # General object observations (if not stealth)
            if not STEALTH_MODE and yolo_detections_this_cycle and (current_time - last_general_speech_time > general_speech_interval):
                objects_to_mention_general = []
                current_active_track_bboxes = {t['id']: t['bbox'] for t in active_trackers if t['active']}
                for det in yolo_detections_this_cycle:
                    if not det['is_priority']: 
                        is_actively_tracked = False
                        for track_id, track_bbox in current_active_track_bboxes.items():
                            track_info = next((t for t in active_trackers if t['id'] == track_id), None)
                            if track_info and track_info['label'] == det['label'] and calculate_iou(det['bbox'], track_bbox) > 0.5:
                                is_actively_tracked = True; break
                        if not is_actively_tracked:
                            objects_to_mention_general.append(f"{det['label']}{' (FZ)' if det['in_focus'] else ''}")
                
                unique_general_mentions = list(set(objects_to_mention_general))
                if unique_general_mentions:
                    if len(unique_general_mentions) == 1: speak(f"Also observed: a {unique_general_mentions[0]}.")
                    else: speak(f"General observation: a " + ", a ".join(unique_general_mentions) + ".")
                last_general_speech_time = current_time
            
            # OCR Text Reporting (if enabled and text found, not in stealth)
            if OCR_ENABLED and ocr_text_this_cycle and not STEALTH_MODE and (current_time - last_ocr_report_time > ocr_report_interval):
                 # Basic report - more sophisticated filtering could be added
                 speak(f"Detected text in focus zone: {ocr_text_this_cycle[:100]}{'...' if len(ocr_text_this_cycle)>100 else ''}") 
                 last_ocr_report_time = current_time


            # --- Status Report ---
            if SYSTEM_STATUS_REPORTING and (current_time - last_status_report_time > status_report_interval):
                tracked_count = sum(1 for t in active_trackers if t['active'])
                status_msg = f"System status: {tracked_count} objects tracked. "
                if STEALTH_MODE: status_msg += "Stealth ON. "
                if LOW_LIGHT_MODE: status_msg += "Low Light ON. "
                if SOUND_DETECTION_MODE: status_msg += "Audio Mon. ON. "
                if OCR_ENABLED: status_msg += "OCR ON."
                speak(status_msg, True, is_priority=False) 
                last_status_report_time = current_time

            # --- Display ---
            if USE_GUI:
                try:
                    # Draw Tracked Objects
                    for track_info in active_trackers:
                        if track_info['active'] and track_info['bbox']: 
                            p1, p2 = (track_info['bbox'][0], track_info['bbox'][1]), (track_info['bbox'][0] + track_info['bbox'][2], track_info['bbox'][1] + track_info['bbox'][3])
                            color = (255, 0, 0); 
                            if track_info.get('is_priority', False): color = (0, 0, 255); 
                            if track_info['in_focus'] and track_info.get('is_priority', False): color = (0, 165, 255); 
                            cv2.rectangle(current_frame_for_drawing, p1, p2, color, 2, 1) 
                            label_text = f"T{track_info['id']}:{track_info['label']}" + ("(FZ)" if track_info['in_focus'] else "") + (f" {track_info.get('direction','')[0].upper()}" if track_info.get('direction') else "")
                            cv2.putText(current_frame_for_drawing, label_text, (p1[0], p1[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
                    # Draw YOLO Detections (not actively tracked)
                    current_active_track_bboxes_for_draw = {t['id']: t['bbox'] for t in active_trackers if t['active']}
                    for det in yolo_detections_this_cycle:
                        is_being_actively_tracked = False
                        for track_id, track_bbox in current_active_track_bboxes_for_draw.items():
                             track_info = next((t for t in active_trackers if t['id'] == track_id), None)
                             if track_info and track_info['label'] == det['label'] and calculate_iou(det['bbox'], track_bbox) > 0.5:
                                 is_being_actively_tracked = True; break
                        if not is_being_actively_tracked:
                            x, y, w, h = det['bbox']; color = (0, 255, 0); 
                            if det['is_priority']: color = (0, 128, 0); 
                            if det['in_focus'] and det['is_priority']: color = (0,200,0); 
                            cv2.rectangle(current_frame_for_drawing, (x,y), (x+w, y+h), color, 2)
                            label_text = f"{det['label']}: {det['confidence']:.2f}" + (" (FZ)" if det['in_focus'] else "")
                            cv2.putText(current_frame_for_drawing, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                    if FOCUS_ZONE_ENABLED:
                        cv2.rectangle(current_frame_for_drawing, (fz_x1, fz_y1), (fz_x2, fz_y2), (0, 255, 255), 1) 
                        cv2.putText(current_frame_for_drawing, "Focus Zone", (fz_x1, fz_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255),1)
                    
                    # Display OCR text (if enabled and found)
                    if OCR_ENABLED and ocr_text_this_cycle:
                         cv2.putText(current_frame_for_drawing, "OCR:"+ocr_text_this_cycle[:30], (10, frame_height - 10), 
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1) # Cyan text at bottom-left


                    title_status = f"S:{'ON' if STEALTH_MODE else 'OFF'}|L:{'ON' if LOW_LIGHT_MODE else 'OFF'}|A:{'ON' if SOUND_DETECTION_MODE else 'OFF'}|R:{'ON' if SYSTEM_STATUS_REPORTING else 'OFF'}|T:{'ON' if OCR_ENABLED else 'OFF'}" # Added OCR status
                    cv2.imshow(f'{AI_NAME_SHORT} Field Ops ({title_status}) - Q:Quit', current_frame_for_drawing)
                    gui_initialized_successfully = True
                except cv2.error as e:
                    if "The function is not implemented" in str(e):
                        if not gui_initialized_successfully: print_opencv_gui_warning(); speak("Warning: Display disabled.",True, is_priority=True)
                        USE_GUI = False 
                    else: print(f"imshow error: {e}"); speak("Display error.",True, is_priority=True); USE_GUI = False 
            
            key_press = cv2.waitKey(1) & 0xFF 
            if key_press == ord('q'): speak("Deactivating system.", True, is_priority=True); break
            elif key_press == ord('s'): STEALTH_MODE = not STEALTH_MODE; speak(f"Stealth Mode {'ENGAGED' if STEALTH_MODE else 'DISENGAGED'}.", True, is_priority=True)
            elif key_press == ord('l'): LOW_LIGHT_MODE = not LOW_LIGHT_MODE; speak(f"Low Light Enhancement {'ACTIVATED' if LOW_LIGHT_MODE else 'DEACTIVATED'}.", True, is_priority=True)
            elif key_press == ord('a'): SOUND_DETECTION_MODE = not SOUND_DETECTION_MODE; speak(f"Simulated Sound Detection {'ENABLED' if SOUND_DETECTION_MODE else 'DISABLED'}.", True, is_priority=True)
            elif key_press == ord('x'): 
                if SOUND_DETECTION_MODE: simulated_sound_event_triggered = True 
                else: speak("Sound detection mode is off.", True)
            elif key_press == ord('r'): SYSTEM_STATUS_REPORTING = not SYSTEM_STATUS_REPORTING; speak(f"System Status Reporting {'ENABLED' if SYSTEM_STATUS_REPORTING else 'DISABLED'}.", True, is_priority=True)
            elif key_press == ord('t'): # Toggle OCR
                OCR_ENABLED = not OCR_ENABLED
                speak(f"OCR Text Detection {'ENABLED' if OCR_ENABLED else 'DISABLED'}.", True, is_priority=True)


    except KeyboardInterrupt: speak("Deactivation by user interrupt.", True, is_priority=True)
    finally:
        print("Shutting down threads...")
        stop_detection_event.set() 
        stop_speech_event.set() 

        if detection_thread.is_alive():
            try: frame_queue.put_nowait(None) 
            except queue.Full: pass
            detection_thread.join(timeout=2.0) # Longer timeout for YOLOv8m
        
        if speech_thread.is_alive():
            try: speech_queue.put_nowait({'text': "Finalizing shutdown.", 'is_priority': False}) 
            except queue.Full: pass
            speech_thread.join(timeout=2) 
        
        if cap: cap.release()
        if gui_initialized_successfully:
            try: cv2.destroyAllWindows()
            except cv2.error: pass 
        print(f"{AI_NAME_LONG}: Systems offline.")

def print_opencv_gui_warning():
    print("\n*** OpenCV GUI Error: Cannot display camera window. Ensure 'opencv-contrib-python' is installed and GUI libs are available. Disabling GUI. ***\n")

if __name__ == '__main__':
    if model is None:
         print("\n--- CRITICAL ERROR: Failed to load the Ultralytics YOLO model. Check previous errors. Exiting. ---")
    elif not CLASS_NAMES:
         print("\n--- CRITICAL ERROR: Failed to load class names. Check '{CLASS_LABELS_FILE}'. Exiting. ---")
    else:
        speak(f"Hello! I am {AI_NAME_LONG}, but you can call me {AI_NAME_SHORT}. System check nominal.", False)
        main_loop()
