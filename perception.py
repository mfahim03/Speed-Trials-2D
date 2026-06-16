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
POLICE_BLUE_BLOB_AREA = 300  # minimum blob area for police car blue section
POLICE_RED_BLOB_AREA  = 200  # minimum blob area for police car red section

COLOR_RANGES = {
    # Green token sprite: light lime green centered near OpenCV HSV H=60.
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


def detect_police_event(frame):
    if frame is None:
        return False

    height, width = frame.shape[:2]

    # The police car spawns on the road ahead. Scan the center of the frame
    # where the car would appear as it approaches. Skip the HUD at top and the player car at bottom.
    roi = frame[int(height * 0.15):int(height * 0.55), int(width * 0.20):int(width * 0.80)]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Police car body has a bright blue half. High saturation (150+) filters out the dark night sky
    # which shares the same hue but is too desaturated to match.
    blue_mask = cv2.inRange(hsv, np.array([100, 150, 80]), np.array([135, 255, 255]))

    # Police car body also has a red half. Red wraps around the hue wheel so we check both ends.
    red_mask1 = cv2.inRange(hsv, np.array([0,   150, 80]), np.array([10,  255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([165, 150, 80]), np.array([179, 255, 255]))
    red_mask  = cv2.bitwise_or(red_mask1, red_mask2)

    # Close small gaps so the car body reads as one solid shape
    kernel = np.ones((5, 5), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    red_mask  = cv2.morphologyEx(red_mask,  cv2.MORPH_CLOSE, kernel)

    # Find contours and check for a solid blob rather than just scattered pixels.
    # The night sky background leaks diffuse blue across many small pixels but never forms
    # one large blob. A real police car body will produce a single large solid blob.
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

    # Red and blue halves of the police car body must be spatially adjacent.
    # False positives (road chevrons + sky) are diagonally across the ROI; a real car's
    # two colour halves sit side-by-side within ~40% of the ROI width.
    return dist < roi.shape[1] * 0.40


def choose_token_action(frame, current_lane, lane_count, police_mode=False):
    # when police_mode is True, red token becomes a target we need to collect, not to avoid
    tokens = []
    for color_name in ('green', 'red', 'yellow'):
        tokens.extend(find_color_tokens(frame, color_name))

    if not tokens:
        # No tokens visible, just cruise and stay in current lane
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

    # police mode: red token is now the goal, not a danger
    # steer toward the red token at full speed to collect it and clear the police event
    if police_mode:
        height, width = frame.shape[:2]
        # exclude road boundary edges (far left/right) and upper half — only real lane tokens remain
        red_targets = [t for t in tokens if t['color'] == 'red' and t['area'] >= DANGER_MIN_AREA
                       and height * 0.30 < t['center_y'] < height * 0.78
                       and width * 0.15 < t['center_x'] < width * 0.73
                       and 0.5 < t['aspect_ratio'] < 2.0]
        if red_targets:
            chosen = max(red_targets, key=lambda t: t['area'])
            target_lane = chosen['lane']
            if target_lane > current_lane:
                steering = 1.0
            elif target_lane < current_lane:
                steering = -1.0
            else:
                steering = 0.0
            debug_frame = frame.copy()
            x, y, w, h = chosen['box']
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 100, 255), 3)
            cv2.putText(debug_frame, f"POLICE MODE: chasing red at lane {target_lane}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
            return steering, 1.0, 'red_target', green_lane, red_lane, target_lane, debug_frame
        # police mode is active but red token is not visible yet, cruise and keep looking
        debug_frame = frame.copy()
        cv2.putText(debug_frame, "POLICE MODE: looking for red token",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
        return 0.0, 0.85, 'police_waiting', green_lane, red_lane, current_lane, debug_frame

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
