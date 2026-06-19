import threading


shared_data = {
    'latest_front_frame': None,
    'latest_back_frame': None,
    'steering_input': 0.0,
    'acceleration_input': 0.0,
    'detected_token': 'none',
    'green_lane': None,
    'red_lane': None,
    'target_lane': 1,
    'event_type': 'none',
    'low_brightness': False,
    'frame_corrupted': False,
    'frame_ok': False,
    'police_active': False,
    'golden_lane_active': False,
    'golden_lane_target': None,
}

data_lock = threading.Lock()
decision_lock = threading.Lock()