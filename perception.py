import cv2
import numpy as np


TOKEN_MIN_AREA = 80
GREEN_MIN_AREA = 100
DANGER_MIN_AREA = 180
PATH_DANGER_BAND = 0.42
TOKEN_ROI_TOP_RATIO = 0.12
LOW_BRIGHTNESS_THRESHOLD = 45
GREYSCALE_SATURATION_THRESHOLD = 20  
TRAILING_MIN_AREA_RATIO = 0.035
TRAILING_CENTER_BAND = 0.70

COLOR_RANGES = {
    'green': [(np.array([35, 30, 30]), np.array([95, 255, 255]))],
    'yellow': [(np.array([18, 80, 80]), np.array([35, 255, 255]))],
    'red': [
        (np.array([0, 80, 80]), np.array([10, 255, 255])),
        (np.array([170, 80, 80]), np.array([179, 255, 255]))
    ]
}


def find_color_tokens(frame, color_name):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    height, width = mask.shape

    for lower, upper in COLOR_RANGES[color_name]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

    mask[:int(height * TOKEN_ROI_TOP_RATIO), :] = 0

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tokens = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < TOKEN_MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h else 0
        fill_ratio = area / (w * h) if w * h else 0
        if aspect_ratio < 0.30 or aspect_ratio > 3.0 or fill_ratio < 0.20:
            continue

        tokens.append({
            'color': color_name,
            'area': area,
            'center_x': x + (w / 2),
            'center_y': y + (h / 2),
            'box': (x, y, w, h),
            'aspect_ratio': aspect_ratio,
            'fill_ratio': fill_ratio
        })

    return tokens


def lane_for_color(tokens, color_name):
    color_tokens = [token for token in tokens if token['color'] == color_name]
    if not color_tokens:
        return None
    return max(color_tokens, key=lambda item: item['area'])['lane']


def analyze_frame_quality(frame):
    if frame is None:
        return False, False, False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    average_brightness = float(np.mean(gray))
    low_brightness = average_brightness < LOW_BRIGHTNESS_THRESHOLD

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    average_saturation = float(np.mean(hsv[:, :, 1]))
    frame_corrupted = average_saturation < GREYSCALE_SATURATION_THRESHOLD

    frame_ok = not low_brightness and not frame_corrupted
    return frame_ok, low_brightness, frame_corrupted


def detect_trailing_car(back_frame):
    frame_ok, low_brightness, _ = analyze_frame_quality(back_frame)
    if not frame_ok:
        return False, low_brightness, back_frame

    height, width = back_frame.shape[:2]
    roi_top = int(height * 0.35)
    roi = back_frame[roi_top:, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    saturated_mask = cv2.inRange(hsv, np.array([0, 45, 45]), np.array([179, 255, 255]))

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 150)
    mask = cv2.bitwise_or(saturated_mask, edges)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    debug_frame = back_frame.copy()
    trailing_detected = False
    min_area = width * height * TRAILING_MIN_AREA_RATIO

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + (w / 2)
        center_offset = abs(center_x - (width / 2)) / (width / 2)
        if center_offset > TRAILING_CENTER_BAND:
            continue

        trailing_detected = True
        cv2.rectangle(debug_frame, (x, y + roi_top), (x + w, y + roi_top + h), (255, 0, 255), 2)

    if trailing_detected:
        cv2.putText(
            debug_frame,
            "TRAILING CAR",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 0, 255),
            2
        )

    return trailing_detected, low_brightness, debug_frame


def choose_token_action(frame, current_lane, lane_count):
    tokens = []
    for color_name in ('green', 'red', 'yellow'):
        tokens.extend(find_color_tokens(frame, color_name))

    if not tokens:
        return 0.0, 1.0, 'none', None, None, current_lane, frame

    height, width = frame.shape[:2]
    lane_left = 0
    lane_right = lane_count - 1

    def clamp_lane(lane):
        return max(lane_left, min(lane_right, lane))

    def lane_from_x(center_x):
        lane_width = width / lane_count
        return clamp_lane(int(center_x / lane_width))

    def safest_lane_away_from(danger_lane):
        lane_scores = {lane: 0 for lane in range(lane_count)}
        lane_scores[danger_lane] -= 100

        for lane in range(lane_count):
            lane_scores[lane] += abs(lane - danger_lane) * 10
            lane_scores[lane] -= abs(lane - current_lane) * 3

        lane_scores[current_lane] += 5
        return max(lane_scores, key=lane_scores.get)

    for token in tokens:
        token['lane'] = lane_from_x(token['center_x'])
        token['center_offset'] = (token['center_x'] - (width / 2)) / (width / 2)

    green_lane = lane_for_color(tokens, 'green')
    red_lane = lane_for_color(tokens, 'red')

    danger_tokens = [
        token for token in tokens
        if token['color'] in ('red', 'yellow') and token['area'] >= DANGER_MIN_AREA
    ]
    green_tokens = [
        token for token in tokens
        if token['color'] == 'green' and token['area'] >= GREEN_MIN_AREA
    ]

    path_dangers = [
        token for token in danger_tokens
        if abs(token['center_offset']) <= PATH_DANGER_BAND or token['lane'] == current_lane
    ]

    if path_dangers:
        chosen = max(path_dangers, key=lambda item: item['area'])
        target_lane = safest_lane_away_from(chosen['lane'])
        acceleration = 0.45
    elif green_tokens:
        chosen = max(green_tokens, key=lambda item: item['area'])
        target_lane = chosen['lane']
        acceleration = 1.0
    elif danger_tokens:
        chosen = max(danger_tokens, key=lambda item: item['area'])
        target_lane = current_lane
        acceleration = 0.70
    else:
        chosen = max(tokens, key=lambda item: item['area'])
        target_lane = current_lane
        acceleration = 0.70

    if target_lane > current_lane:
        steering = 1.0
    elif target_lane < current_lane:
        steering = -1.0
    else:
        steering = 0.0

    debug_frame = frame.copy()
    for token in tokens:
        x, y, w, h = token['box']
        box_color = (0, 255, 0)
        if token['color'] == 'red':
            box_color = (0, 0, 255)
        elif token['color'] == 'yellow':
            box_color = (0, 255, 255)
        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), box_color, 2)

    x, y, w, h = chosen['box']
    cv2.putText(
        debug_frame,
        f"{chosen['color']} lane={chosen['lane']} target={target_lane} steer={steering:.0f}",
        (x, max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    return steering, acceleration, chosen['color'], green_lane, red_lane, target_lane, debug_frame
