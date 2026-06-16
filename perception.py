import cv2
import numpy as np


TOKEN_MIN_AREA = 80          # was 300 — was filtering green tokens before GREEN_MIN_AREA check
GREEN_MIN_AREA = 100         # changed from 80 (was 220 originally) to match spec
DANGER_MIN_AREA = 120        # was 180 — detect red/yellow hazards sooner
PATH_DANGER_BAND = 0.65      # was 0.42 — wider band, avoids more off-center dangers
TOKEN_ROI_TOP_RATIO = 0.10   # was 0.12 — look slightly higher up the frame
LOW_BRIGHTNESS_THRESHOLD = 45
GREYSCALE_SATURATION_THRESHOLD = 20
TRAILING_MIN_AREA_RATIO = 0.035
TRAILING_CENTER_BAND = 0.70

COLOR_RANGES = {
    # Tuned green: wider hue (30–95), lower S/V min (20) to catch darker/paler greens
    'green': [(np.array([30, 20, 20]), np.array([95, 255, 255]))],
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

        # Relaxed shape filter — green tokens far away can look squarish or irregular
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
    height = frame.shape[0]
    sample_region = frame[:int(height * 0.7), :]  # exclude bottom 30% (close-up token area)

    gray = cv2.cvtColor(sample_region, cv2.COLOR_BGR2GRAY)
    average_brightness = float(np.mean(gray))
    low_brightness = average_brightness < LOW_BRIGHTNESS_THRESHOLD

    hsv = cv2.cvtColor(sample_region, cv2.COLOR_BGR2HSV)
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
        # No tokens visible — cruise at moderate speed, stay in lane
        return 0.0, 0.6, 'none', None, None, current_lane, frame

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

    # Wider PATH_DANGER_BAND means more off-center hazards trigger avoidance
    path_dangers = [
        token for token in danger_tokens
        if abs(token['center_offset']) <= PATH_DANGER_BAND or token['lane'] == current_lane
    ]

    # All lanes currently occupied by ANY danger token (not just path dangers) —
    # used to make sure we never steer toward green through a danger lane.
    danger_lanes = {token['lane'] for token in danger_tokens}

    def safe_green_tokens():
        """Green tokens whose lane does not overlap any danger token's lane."""
        return [t for t in green_tokens if t['lane'] not in danger_lanes]

    # -------------------------------------------------------
    # Priority 1: Danger is in our path — SLOW DOWN and dodge
    # -------------------------------------------------------
    if path_dangers:
        chosen = max(path_dangers, key=lambda item: item['area'])
        safe_greens = safe_green_tokens()

        # Only steer toward green if its lane is clear of ALL danger tokens
        if safe_greens:
            green_chosen = max(safe_greens, key=lambda item: item['area'])
            target_lane = green_chosen['lane']
            chosen = green_chosen
            acceleration = 0.5   # slow enough to lane-change cleanly, green is close
        else:
            target_lane = safest_lane_away_from(chosen['lane'])
            acceleration = 0.35  # slow — give time for lane tap to complete before impact

    # -------------------------------------------------------
    # Priority 2: Green visible, no immediate path danger — GO GET IT
    # (but still never steer into a lane that has a danger token sitting in it)
    # -------------------------------------------------------
    elif green_tokens:
        safe_greens = safe_green_tokens()
        if safe_greens:
            chosen = max(safe_greens, key=lambda item: item['area'])
            target_lane = chosen['lane']
            acceleration = 1.0   # full speed toward green
        else:
            # every visible green sits in a danger lane — fall back to avoiding danger instead
            chosen = max(danger_tokens, key=lambda item: item['area'])
            target_lane = safest_lane_away_from(chosen['lane'])
            acceleration = 0.4

    # -------------------------------------------------------
    # Priority 3: Danger exists but not in our path — slow slightly, hold lane
    # -------------------------------------------------------
    elif danger_tokens:
        chosen = max(danger_tokens, key=lambda item: item['area'])
        target_lane = current_lane
        acceleration = 0.5   # slow down cautiously, stay in lane

    # -------------------------------------------------------
    # Priority 4: Unknown tokens — hold lane, moderate speed
    # -------------------------------------------------------
    else:
        chosen = max(tokens, key=lambda item: item['area'])
        target_lane = current_lane
        acceleration = 0.6

    if target_lane > current_lane:
        steering = 1.0
    elif target_lane < current_lane:
        steering = -1.0
    else:
        steering = 0.0

    # Debug overlay
    debug_frame = frame.copy()
    for token in tokens:
        x, y, w, h = token['box']
        if token['color'] == 'green':
            box_color = (0, 255, 0)
        elif token['color'] == 'red':
            box_color = (0, 0, 255)
        else:
            box_color = (0, 255, 255)
        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), box_color, 2)

    x, y, w, h = chosen['box']
    cv2.putText(
        debug_frame,
        f"{chosen['color']} lane={chosen['lane']} target={target_lane} steer={steering:.0f} accel={acceleration:.2f}",
        (x, max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    return steering, acceleration, chosen['color'], green_lane, red_lane, target_lane, debug_frame