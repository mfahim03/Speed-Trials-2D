import socket
import threading
import struct
import cv2
import numpy as np
import time
import keyboard
import select
import ctypes

from perception import analyze_frame_quality, choose_token_action, detect_trailing_car
from rt_shared import data_lock, decision_lock, shared_data

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
CAMERA_HOST = '127.0.0.1'
FRONT_CAMERA_PORT = 8080
BACK_CAMERA_PORT = 8082
CONTROL_HOST = '127.0.0.1'
CONTROL_PORT = 8081

is_running = True

# Manual control tuning
AUTO_DRIVE_ENABLED = True
LOW_BRIGHTNESS_ACCELERATION = -1.0
STEERING_TAP_DURATION = 0.22
STEERING_TAP_COOLDOWN = 0.45
LANE_COUNT = 4
LANE_LEFT = 0
LANE_RIGHT = LANE_COUNT - 1
current_lane = 1
last_left_pressed = False
last_right_pressed = False
steering_tap_until = 0.0
steering_tap_value = 0.0
auto_next_tap_time = 0.0
trailing_tap_cooldown = 0.0

# ---------------------------------------------------------
# Real-Time Scheduling Framework (Do not change this in your code)
# ---------------------------------------------------------
class TaskPriority:
    HIGH = 1
    MEDIUM = 2
    LOW = 3

class RTTask(threading.Thread):
    """
    Real-Time Task implementing:
    - Concurrency (inherits threading.Thread)
    - Task Period (enforced in run loop)
    - Task Priority (logical priority assigned)
    """
    def __init__(self, name, period, priority, execute_func):
        super().__init__()
        self.name = name
        self.period = period
        self.priority = priority
        self.execute_func = execute_func
        self.daemon = True

    def run(self):
        print(f"[{self.name}] Started | Period: {self.period}s | Priority: {self.priority}")
        try:
            handle = ctypes.windll.kernel32.GetCurrentThread()
            if self.priority == TaskPriority.HIGH:
                ctypes.windll.kernel32.SetThreadPriority(handle, 2)
            elif self.priority == TaskPriority.MEDIUM:
                ctypes.windll.kernel32.SetThreadPriority(handle, 0)
            elif self.priority == TaskPriority.LOW:
                ctypes.windll.kernel32.SetThreadPriority(handle, -2)
        except Exception as e:
            pass

        while is_running:
            start_time = time.time()
            self.execute_func()
            exec_time = time.time() - start_time
            sleep_time = self.period - exec_time
            
            if sleep_time > 0:
                time.sleep(sleep_time)

# ---------------------------------------------------------
# Network Connection Setup (Do not change this in your code)
# ---------------------------------------------------------
front_camera_sock = None
back_camera_sock = None
control_conn = None

def setup_cameras():
    global front_camera_sock, back_camera_sock
    
    print("Connecting to Cameras...")
    front_connected = False
    back_connected = False
    
    while is_running and not (front_connected and back_connected):
        if not front_connected:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect((CAMERA_HOST, FRONT_CAMERA_PORT))
                front_camera_sock = s
                print("Connected to Front Camera successfully.")
                front_connected = True
            except Exception:
                pass
                
        if not back_connected:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect((CAMERA_HOST, BACK_CAMERA_PORT))
                back_camera_sock = s
                print("Connected to Back Camera successfully.")
                back_connected = True
            except Exception:
                pass
                
        if not (front_connected and back_connected):
            time.sleep(1)

def setup_control_server():
    global control_conn
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((CONTROL_HOST, CONTROL_PORT))
    server_sock.listen()
    server_sock.settimeout(1.0)
    print(f"Control server listening on {CONTROL_HOST}:{CONTROL_PORT}")
    
    while is_running:
        try:
            conn, addr = server_sock.accept()
            print(f"Control client connected from {addr}")
            control_conn = conn
            break
        except socket.timeout:
            continue

# ---------------------------------------------------------
# Task Implementations (This is where you write your tasks)
# ---------------------------------------------------------

def read_single_camera(sock, window_name, data_key):
    #This function reads the latest frame from the camera socket and stores it in the shared data
    if sock is None:
        return
        
    try:
        latest_frame_data = None
        sock.settimeout(None)
        length_bytes = sock.recv(4)
        if not length_bytes:
            return
            
        image_length = int.from_bytes(length_bytes, 'little')
        received_bytes = b''
        while len(received_bytes) < image_length and is_running:
            packet = sock.recv(image_length - len(received_bytes))
            if not packet:
                break
            received_bytes += packet
            
        if len(received_bytes) == image_length:
            latest_frame_data = received_bytes
            
        while is_running:
            readable, _, _ = select.select([sock], [], [], 0.0)
            if not readable:
                break
                
            sock.settimeout(1.0)
            length_bytes = sock.recv(4)
            if not length_bytes:
                return
            image_length = int.from_bytes(length_bytes, 'little')
            received_bytes = b''
            while len(received_bytes) < image_length and is_running:
                packet = sock.recv(image_length - len(received_bytes))
                if not packet:
                    break
                received_bytes += packet
                
            if len(received_bytes) == image_length:
                latest_frame_data = received_bytes
                
        if latest_frame_data is not None:
            np_arr = np.frombuffer(latest_frame_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                with data_lock:
                    shared_data[data_key] = frame
                
                # You may disable this if you don't need to display the frames / This could effect the fps
                frame_resized = cv2.resize(frame, (640, 480))
                cv2.imshow(window_name, frame_resized)
                cv2.waitKey(1)
                
    except Exception as e:
        pass

def read_front_camera_task():
    read_single_camera(front_camera_sock, "Front Camera", 'latest_front_frame')

def read_back_camera_task():
    read_single_camera(back_camera_sock, "Back Camera", 'latest_back_frame')

# Image processing code to decide how to control the car
def clamp_lane(lane):
    return max(LANE_LEFT, min(LANE_RIGHT, lane))

def steering_towards_lane(target_lane):
    if target_lane > current_lane:
        return 1.0
    if target_lane < current_lane:
        return -1.0
    return 0.0

def processing_task():
    #This is where you write your image processing code to decide how to control the car
    #You can use libraries like OpenCV to process the image
    #There is no limtation to the complexity of the processing task, you can use any libraries you want
    #Remember to use the shared_data to get the latest frame
    with data_lock:
        front_frame = shared_data['latest_front_frame']
        back_frame = shared_data['latest_back_frame']
    
    if front_frame is None:
        with decision_lock:
            shared_data['frame_ok'] = False
        return

    front_ok, front_low_brightness, front_corrupted = analyze_frame_quality(front_frame)
    if not front_ok:
        with decision_lock:
            shared_data['low_brightness'] = front_low_brightness
            shared_data['frame_corrupted'] = front_corrupted
            shared_data['frame_ok'] = False

            if front_low_brightness:
                shared_data['steering_input'] = 0.0
                shared_data['acceleration_input'] = LOW_BRIGHTNESS_ACCELERATION
                shared_data['detected_token'] = 'none'
                shared_data['event_type'] = 'low_brightness'
            else:  # frame_corrupted
                shared_data['steering_input'] = 0.0
                shared_data['acceleration_input'] = 0.7
                shared_data['target_lane'] = current_lane
                shared_data['detected_token'] = 'unknown'
                shared_data['event_type'] = 'frame_corrupted'
        return

    steering_input, acceleration_input, detected_token, green_lane, red_lane, target_lane, debug_frame = choose_token_action(
        front_frame,
        current_lane,
        LANE_COUNT
    )
    trailing_detected, back_low_brightness, back_debug_frame = detect_trailing_car(back_frame)
    if trailing_detected:
        event_type = 'trailing'
        acceleration_input = 1.0
    else:
        event_type = detected_token

    low_brightness = front_low_brightness
    frame_ok = front_frame is not None and back_frame is not None and not low_brightness
    if low_brightness:
        steering_input = 0.0
        acceleration_input = LOW_BRIGHTNESS_ACCELERATION
        event_type = 'low_brightness'

    with decision_lock:
        shared_data['steering_input'] = steering_input
        shared_data['acceleration_input'] = acceleration_input
        shared_data['detected_token'] = detected_token
        shared_data['green_lane'] = green_lane
        shared_data['red_lane'] = red_lane
        shared_data['target_lane'] = target_lane
        shared_data['event_type'] = event_type
        shared_data['low_brightness'] = low_brightness
        shared_data['frame_ok'] = frame_ok

    cv2.imshow("Token Detection", cv2.resize(debug_frame, (640, 480)))
    if back_debug_frame is not None:
        cv2.imshow("Trailing Detection", cv2.resize(back_debug_frame, (640, 480)))
    cv2.waitKey(1)

def send_controls_task():
    #This is where you send the control commands to the car using the control_conn
    global control_conn, is_running
    global last_left_pressed, last_right_pressed, steering_tap_until, steering_tap_value
    global auto_next_tap_time, current_lane
    global trailing_tap_cooldown
    if control_conn is None:
        return
    
    #these are the variables used to control the car
    #steering_input: -1.0 to 1.0 (left to right)
    #acceleration_input: -1.0 to 1.0 (reverse to forward)
    # W / Up: accelerate, S / Down: reverse/brake
    # A / Left and D / Right send short steering taps for lane switching.
    manual_acceleration = None
    if keyboard.is_pressed('w') or keyboard.is_pressed('up'):
        manual_acceleration = 1.0
    elif keyboard.is_pressed('s') or keyboard.is_pressed('down'):
        manual_acceleration = -1.0

    now = time.time()
    left_pressed = keyboard.is_pressed('a') or keyboard.is_pressed('left')
    right_pressed = keyboard.is_pressed('d') or keyboard.is_pressed('right')

    if left_pressed and not last_left_pressed:
        steering_tap_value = -1.0
        steering_tap_until = now + STEERING_TAP_DURATION
        current_lane = clamp_lane(current_lane - 1)
    elif right_pressed and not last_right_pressed:
        steering_tap_value = 1.0
        steering_tap_until = now + STEERING_TAP_DURATION
        current_lane = clamp_lane(current_lane + 1)

    last_left_pressed = left_pressed
    last_right_pressed = right_pressed

    with decision_lock:
        auto_steering = shared_data['steering_input']
        auto_acceleration = shared_data['acceleration_input']
        target_lane = shared_data['target_lane']
        event_type = shared_data['event_type']
        
        # MUNA — Trailing car auto lane switch
        if event_type == 'trailing' and now >= trailing_tap_cooldown:
              if current_lane <= 1:
                 # left side — tap right
                 steering_tap_value = 1.0
                 steering_tap_until = now + STEERING_TAP_DURATION
                 current_lane = clamp_lane(current_lane + 1)
              else:
                 # right side — tap left
                 steering_tap_value = -1.0
                 steering_tap_until = now + STEERING_TAP_DURATION
                 current_lane = clamp_lane(current_lane - 1)
                 trailing_tap_cooldown = now + 2.0  # 2 second cooldown

        # MUNA — Police mode acceleration
        if event_type == 'police':
            auto_acceleration = 1.0

    if AUTO_DRIVE_ENABLED and auto_steering != 0.0 and now >= auto_next_tap_time:
        steering_tap_value = steering_towards_lane(target_lane)
        steering_tap_until = now + STEERING_TAP_DURATION
        auto_next_tap_time = now + STEERING_TAP_COOLDOWN
        current_lane = clamp_lane(current_lane + int(steering_tap_value))

    if now < steering_tap_until:
        steering_input = steering_tap_value
    else:
        steering_input = 0.0

    if manual_acceleration is not None:
        acceleration_input = manual_acceleration
    elif AUTO_DRIVE_ENABLED:
        acceleration_input = auto_acceleration
    else:
        acceleration_input = 0.0

    if keyboard.is_pressed('q'):
        is_running = False

    try:
        # Pack and send the control command
        data = struct.pack('ff', steering_input, acceleration_input)
        control_conn.sendall(data)
    except Exception as e:
        print(f"Control send error: {e}")
        control_conn = None


# ---------------------------------------------------------
# Main (Scheduler Initialization)
# ---------------------------------------------------------
if __name__ == '__main__':
    print("Initializing RTSE Sample Drive...")
    
    # Initialize network connections
    threading.Thread(target=setup_control_server, daemon=True).start()
    threading.Thread(target=setup_cameras, daemon=True).start()
    
    print("\n--- Starting Real-Time Tasks (awaiting connections dynamically) ---\n")
    
    # This is where you define tasks with explicit Scheduling parameters (Concurrency, Priority, Period)
    # Period refers to the period of execution of the task in seconds
    # Priority refers to the priority of the task, higher priority means higher priority
    # Concurrency refers to the number of instances of the task that can run at the same time
    t_front_camera = RTTask("ReadFrontCamera", period=0.005, priority=TaskPriority.HIGH, execute_func=read_front_camera_task)
    t_back_camera = RTTask("ReadBackCamera", period=0.005, priority=TaskPriority.HIGH, execute_func=read_back_camera_task)
    t_processing = RTTask("Processing", period=0.005, priority=TaskPriority.MEDIUM, execute_func=processing_task)
    t_controls = RTTask("SendControls", period=0.005, priority=TaskPriority.HIGH, execute_func=send_controls_task)
    
    # Start tasks to run concurrently
    t_front_camera.start()
    t_back_camera.start()
    t_processing.start()
    t_controls.start()
    
    try:
        # You need this to keep the main thread alive, otherwise the program will exit immediately
        while is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detected. Stopping system...")
        is_running = False

    # This is to make sure that the tasks are terminated cleanly
    t_front_camera.join()
    t_back_camera.join()
    t_processing.join()
    t_controls.join()
    
    # This is to close all the connections
    if front_camera_sock:
        front_camera_sock.close()
    if back_camera_sock:
        back_camera_sock.close()
    if control_conn:
        control_conn.close()
    cv2.destroyAllWindows()
    print("System terminated cleanly.")
