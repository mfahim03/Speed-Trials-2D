import cv2
import numpy as np

#cinta datang
TOKEN_MIN_AREA = 80
GREEN_MIN_AREA = 100
DANGER_MIN_AREA = 120
PATH_DANGER_BAND = 0.65
TOKEN_ROI_TOP_RATIO = 0.10
LOW_BRIGHTNESS_THRESHOLD = 45
GREYSCALE_SATURATION_THRESHOLD = 20
TRAILING_MIN_AREA_RATIO = 0.035
TRAILING_CENTER_BAND = 0.70
POLICE_BLUE_BLOB_AREA = 300
POLICE_RED_BLOB_AREA  = 200

COLOR_RANGES = {
    'green': [(np.array([50, 70, 120]), np.array([70, 255, 255]))],
    'yellow': [(np.array([13, 55, 100]), np.array([36, 255, 255]))],
    'red': [
        (np.array([0, 70, 120]), np.array([8, 255, 255])),
        (np.array([175, 70, 120]), np.array([179, 255, 255]))
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
        return False, False, False, 0.0

    height = frame.shape[0]
    sample_region = frame[:int(height * 0.7), :]

    gray = cv2.cvtColor(sample_region, cv2.COLOR_BGR2GRAY)
    average_brightness = float(np.mean(gray))
    low_brightness = average_brightness < LOW_BRIGHTNESS_THRESHOLD

    hsv = cv2.cvtColor(sample_region, cv2.COLOR_BGR2HSV)
    average_saturation = float(np.mean(hsv[:, :, 1]))
    frame_corrupted = average_saturation < GREYSCALE_SATURATION_THRESHOLD

    frame_ok = not low_brightness and not frame_corrupted
    return frame_ok, low_brightness, frame_corrupted, average_brightness


def detect_trailing_car(back_frame):
    frame_ok, low_brightness, _, _ = analyze_frame_quality(back_frame)
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
        cv2.putText(debug_frame, "TRAILING CAR", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)

    return trailing_detected, low_brightness, debug_frame


def detect_police_event(frame):
    if frame is None:
        return False

    height, width = frame.shape[:2]
    roi = frame[int(height * 0.15):int(height * 0.55), int(width * 0.20):int(width * 0.80)]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    blue_mask = cv2.inRange(hsv, np.array([100, 150, 80]), np.array([135, 255, 255]))
    red_mask1 = cv2.inRange(hsv, np.array([0,   150, 80]), np.array([10,  255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([165, 150, 80]), np.array([179, 255, 255]))
    red_mask  = cv2.bitwise_or(red_mask1, red_mask2)

    kernel = np.ones((5, 5), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    red_mask  = cv2.morphologyEx(red_mask,  cv2.MORPH_CLOSE, kernel)

    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_contours,  _ = cv2.findContours(red_mask,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blue_blobs = [c for c in blue_contours if cv2.contourArea(c) >= POLICE_BLUE_BLOB_AREA]
    red_blobs  = [c for c in red_contours  if cv2.contourArea(c) >= POLICE_RED_BLOB_AREA]

    if not blue_blobs or not red_blobs:
        return False

    largest_blue = max(blue_blobs, key=cv2.contourArea)
    largest_red  = max(red_blobs,  key=cv2.contourArea)

    bx, by, bw, bh = cv2.boundingRect(largest_blue)
    rx, ry, rw, rh = cv2.boundingRect(largest_red)
    dist = (((bx + bw / 2) - (rx + rw / 2)) ** 2 + ((by + bh / 2) - (ry + rh / 2)) ** 2) ** 0.5

    return dist < roi.shape[1] * 0.40


def detect_golden_lane(frame):
    if frame is None:
        return False, None

    height, width = frame.shape[:2]
    # Text appears just below the timer, top center
    roi = frame[int(height * 0.08):int(height * 0.18), int(width * 0.15):int(width * 0.85)]

    # Orange/yellow text color in HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    orange_mask = cv2.inRange(hsv, np.array([10, 150, 150]), np.array([25, 255, 255]))
    orange_pixels = cv2.countNonZero(orange_mask)

    if orange_pixels < 30:
        return False, None

    # Try to read lane number using OCR
    try:
        import pytesseract
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresh, config='--psm 7 -c tessedit_char_whitelist=LANEalgreen1234 ')
        text = text.upper()
        print(f"[GOLDEN LANE] OCR text: {text}")
        for ch in '1234':
            if f'LANE {ch}' in text or f'LANE{ch}' in text:
                return True, int(ch) - 1  # convert to 0-indexed
    except Exception:
        pass

    # Fallback: orange pixels detected but couldn't read number
    return True, None


def choose_token_action(frame, current_lane, lane_count, police_mode=False):
    tokens = []
    for color_name in ('green', 'red', 'yellow'):
        tokens.extend(find_color_tokens(frame, color_name))

    if not tokens:
        return 0.0, 0.6, 'none', None, None, current_lane, frame

    height, width = frame.shape[:2]
    lane_left = 0
    lane_right = lane_count - 1
    lane_width = width / lane_count

    REACT_Y_THRESHOLD = height * 0.50
    LANE_TOLERANCE = lane_width * 0.45

    def clamp_lane(lane):
        return max(lane_left, min(lane_right, lane))

    def lane_from_x(center_x):
        return clamp_lane(int(center_x / lane_width))

    def lane_center_x(lane):
        return (lane + 0.5) * lane_width

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
        token['lane_offset'] = abs(token['center_x'] - lane_center_x(current_lane))
        token['close_enough'] = token['center_y'] >= REACT_Y_THRESHOLD

    green_lane = lane_for_color(tokens, 'green')
    red_lane = lane_for_color(tokens, 'red')

    if police_mode:
        red_targets = [t for t in tokens if t['color'] == 'red' and t['area'] >= DANGER_MIN_AREA
                       and height * 0.30 < t['center_y'] < height * 0.78
                       and width * 0.15 < t['center_x'] < width * 0.73
                       and 0.5 < t['aspect_ratio'] < 2.0]
        if red_targets:
            chosen = max(red_targets, key=lambda t: t['area'])
            target_lane = chosen['lane']
            steering = 1.0 if target_lane > current_lane else (-1.0 if target_lane < current_lane else 0.0)
            debug_frame = frame.copy()
            x, y, w, h = chosen['box']
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 100, 255), 3)
            cv2.putText(debug_frame, f"POLICE MODE: chasing red lane {target_lane}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
            return steering, 1.0, 'red_target', green_lane, red_lane, target_lane, debug_frame
        debug_frame = frame.copy()
        cv2.putText(debug_frame, "POLICE MODE: looking for red token",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
        return 0.0, 0.85, 'police_waiting', green_lane, red_lane, current_lane, debug_frame

    # Danger = close enough AND within lane tolerance of current lane center
    danger_tokens = [
        t for t in tokens
        if t['color'] in ('red', 'yellow')
        and t['area'] >= DANGER_MIN_AREA
        and t['close_enough']
        and abs(t['center_x'] - lane_center_x(current_lane)) <= LANE_TOLERANCE
    ]

    # Reds nearby even outside our lane — used to block green chasing
    nearby_reds = [
        t for t in tokens
        if t['color'] == 'red'
        and t['area'] >= DANGER_MIN_AREA
        and t['close_enough']
        and abs(t['center_x'] - lane_center_x(current_lane)) <= lane_width * 1.2
    ]

    green_tokens = [
        t for t in tokens
        if t['color'] == 'green' and t['area'] >= GREEN_MIN_AREA
    ]

    danger_lanes = {t['lane'] for t in danger_tokens}

    def safe_green_tokens():
        return [t for t in green_tokens if t['lane'] not in danger_lanes]

    # Debug frame
    debug_frame = frame.copy()
    cx = int(lane_center_x(current_lane))
    cv2.line(debug_frame, (cx, 0), (cx, height), (0, 255, 0), 2)
    cv2.line(debug_frame, (0, int(REACT_Y_THRESHOLD)), (width, int(REACT_Y_THRESHOLD)), (255, 255, 0), 1)

    # -------------------------------------------------------
    # Priority 1: Danger in our lane — dodge
    # -------------------------------------------------------
    if danger_tokens:
        chosen = max(danger_tokens, key=lambda t: t['area'])
        safe_greens = safe_green_tokens()
        if safe_greens:
            green_chosen = max(safe_greens, key=lambda t: t['area'])
            target_lane = green_chosen['lane']
            chosen = green_chosen
            acceleration = 0.5
        else:
            target_lane = safest_lane_away_from(chosen['lane'])
            acceleration = 0.35

    # -------------------------------------------------------
    # Priority 2: Green visible, no danger in our lane
    # -------------------------------------------------------
    elif green_tokens:
        safe_greens = safe_green_tokens()
        if safe_greens:
            # pick closest green (highest center_y)
            closest_green = max(safe_greens, key=lambda t: t['center_y'])

            # check if any red is closer than the green (between us and green)
            blocking_reds = [
                t for t in tokens
                if t['color'] == 'red'
                and t['area'] >= DANGER_MIN_AREA
                and t['center_y'] > closest_green['center_y']
            ]

            if blocking_reds:
                # red is closer than green — dodge red first
                chosen = max(blocking_reds, key=lambda t: t['area'])
                target_lane = safest_lane_away_from(chosen['lane'])
                acceleration = 0.4
            else:
                chosen = closest_green
                target_lane = chosen['lane']
                # slow down if nearby reds even in adjacent lanes
                acceleration = 0.7 if nearby_reds else 1.0
        else:
            # all greens blocked by danger lanes
            chosen = max(tokens, key=lambda t: t['area'])
            target_lane = safest_lane_away_from(chosen['lane'])
            acceleration = 0.4

    # -------------------------------------------------------
    # Priority 3: No green, danger exists but not in our lane
    # -------------------------------------------------------
    elif danger_tokens or nearby_reds:
        all_dangers = danger_tokens if danger_tokens else nearby_reds
        chosen = max(all_dangers, key=lambda t: t['area'])
        target_lane = current_lane
        acceleration = 0.5

    # -------------------------------------------------------
    # Priority 4: Nothing relevant — cruise
    # -------------------------------------------------------
    else:
        chosen = max(tokens, key=lambda t: t['area'])
        target_lane = current_lane
        acceleration = 0.6

    steering = 1.0 if target_lane > current_lane else (-1.0 if target_lane < current_lane else 0.0)

    for token in tokens:
        x, y, w, h = token['box']
        color = (0, 255, 0) if token['color'] == 'green' else (0, 0, 255) if token['color'] == 'red' else (0, 255, 255)
        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), color, 2)

    x, y, w, h = chosen['box']
    cv2.putText(debug_frame,
        f"{chosen['color']} lane={chosen['lane']} target={target_lane} steer={steering:.0f} accel={acceleration:.2f}",
        (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return steering, acceleration, chosen['color'], green_lane, red_lane, target_lane, debug_frame