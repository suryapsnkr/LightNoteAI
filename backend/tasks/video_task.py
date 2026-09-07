import cv2
import os
import re
import numpy as np
from celery import Task
from tasks.celery_app import celery_app
from app.config import settings
from ultralytics import YOLO


# Load the SEGMENTATION model
yolo_model = YOLO("yolov8n-seg.pt")


def parse_prompt(prompt_text: str) -> dict:
    """
    Extract operation, target object, and replacement directly
    from the user's prompt.

    Examples:
        "remove the bottle"
        -> {
            "operation": "remove",
            "target": "bottle",
            "replacement": None
        }

        "replace the bottle with a flower"
        -> {
            "operation": "replace",
            "target": "bottle",
            "replacement": "flower"
        }

        "change the text on the bottle"
        -> {
            "operation": "change_text",
            "target": "bottle",
            "replacement": None
        }
    """

    prompt = prompt_text.lower().strip()

    operation = "replace"
    target = "object"
    replacement = None

    # ---------------------------------------------------------
    # REMOVE
    # ---------------------------------------------------------
    match = re.search(
        r"\bremove\s+(?:the\s+)?(.+?)(?:\s+from\s+.*)?$",
        prompt
    )

    if match:
        operation = "remove"
        target = match.group(1).strip()

        return {
            "operation": operation,
            "target": target,
            "replacement": None
        }

    # ---------------------------------------------------------
    # REPLACE
    # ---------------------------------------------------------
    match = re.search(
        r"\breplace\s+(?:the\s+)?(.+?)\s+\bwith\s+(.+)$",
        prompt
    )

    if match:
        operation = "replace"
        target = match.group(1).strip()
        replacement = match.group(2).strip()

        return {
            "operation": operation,
            "target": target,
            "replacement": replacement
        }

    # ---------------------------------------------------------
    # CHANGE TEXT
    # ---------------------------------------------------------
    match = re.search(
        r"\bchange\s+(?:the\s+)?text(?:\s+(?:on|in|of)\s+(?:the\s+)?(.+))?$",
        prompt
    )

    if match:
        operation = "change_text"

        if match.group(1):
            target = match.group(1).strip()
        else:
            target = "object"

        return {
            "operation": operation,
            "target": target,
            "replacement": None
        }

    # ---------------------------------------------------------
    # GENERIC "CHANGE X TO Y"
    # ---------------------------------------------------------
    match = re.search(
        r"\bchange\s+(?:the\s+)?(.+?)\s+\bto\s+(.+)$",
        prompt
    )

    if match:
        operation = "replace"
        target = match.group(1).strip()
        replacement = match.group(2).strip()

        return {
            "operation": operation,
            "target": target,
            "replacement": replacement
        }

    # ---------------------------------------------------------
    # FALLBACK
    # ---------------------------------------------------------
    print(f"⚠️ Could not fully parse prompt: {prompt_text}")

    return {
        "operation": operation,
        "target": target,
        "replacement": replacement
    }


def detect_object_yolo(frame, target_text: str) -> dict:
    """
    Runs YOLO segmentation on a frame.

    Returns:
        {
            'x1': int,
            'y1': int,
            'x2': int,
            'y2': int,
            'mask': numpy.ndarray
        }

    The target comes directly from the user prompt.
    """

    results = yolo_model(
        frame,
        conf=0.1,
        verbose=False
    )

    if not results:
        return None

    detections = []

    for r in results:

        if r.boxes is None:
            continue

        for i, box in enumerate(r.boxes):

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            conf = float(box.conf[0])
            cls = int(box.cls[0])
            name = yolo_model.names[cls]

            # -------------------------------------------------
            # Get segmentation mask
            # -------------------------------------------------
            mask = None

            if (
                r.masks is not None
                and len(r.masks.data) > i
            ):
                mask_np = (
                    r.masks.data[i]
                    .cpu()
                    .numpy()
                )

                # Resize mask to frame dimensions
                if mask_np.shape != (
                    frame.shape[0],
                    frame.shape[1]
                ):
                    mask_np = cv2.resize(
                        mask_np,
                        (
                            frame.shape[1],
                            frame.shape[0]
                        )
                    )

                mask = (
                    mask_np > 0.5
                ).astype(np.uint8)

            detections.append({
                "name": name,
                "confidence": conf,
                "box": (x1, y1, x2, y2),
                "mask": mask
            })

    if not detections:
        return None

    # ---------------------------------------------------------
    # Match target from prompt against YOLO class
    # ---------------------------------------------------------

    target_lower = target_text.lower().strip()

    matches = []

    for detection in detections:

        detected_name = detection["name"].lower()

        # Direct match:
        # "bottle" -> "bottle"
        #
        # Partial match:
        # "cell phone" -> "cell phone"
        #
        # Also allows:
        # "a bottle" -> "bottle"
        clean_target = re.sub(
            r"^(a|an|the)\s+",
            "",
            target_lower
        )

        if (
            clean_target == detected_name
            or clean_target in detected_name
            or detected_name in clean_target
        ):
            matches.append(detection)

    if matches:

        # Pick highest confidence detection
        best = max(
            matches,
            key=lambda x: x["confidence"]
        )

        name = best["name"]
        conf = best["confidence"]

        x1, y1, x2, y2 = best["box"]
        mask = best["mask"]

        print(
            f"✅ Detected '{name}' "
            f"for target '{target_text}' "
            f"with confidence {conf:.2f}"
        )

        return {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "mask": mask
        }

    # ---------------------------------------------------------
    # Nothing matched
    # ---------------------------------------------------------

    detected_classes = list(
        set(
            d["name"]
            for d in detections
        )
    )

    print(
        f"⚠️ No object matching "
        f"'{target_text}' found."
    )

    print(
        f"   YOLO detected: {detected_classes}"
    )

    return None


@celery_app.task(bind=True)
def process_video_task(
    self,
    video_path: str,
    prompt: str,
    ref_path: str = None
):
    # ---------------------------------------------------------
    # Parse prompt
    # ---------------------------------------------------------

    parsed = parse_prompt(prompt)

    target = parsed.get(
        "target",
        "object"
    )

    operation = parsed.get(
        "operation",
        "replace"
    )

    replacement = parsed.get(
        "replacement"
    )

    print(
        f"📝 Prompt: {prompt}"
    )

    print(
        f"📝 Parsed:"
        f" operation={operation},"
        f" target='{target}',"
        f" replacement='{replacement}'"
    )

    # ---------------------------------------------------------
    # Open video
    # ---------------------------------------------------------

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = int(
        cap.get(cv2.CAP_PROP_FPS)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    # Prevent division by zero
    if fps <= 0:
        fps = 30

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    os.makedirs(
        settings.OUTPUT_DIR,
        exist_ok=True
    )

    out_path = os.path.join(
        settings.OUTPUT_DIR,
        f"output_{self.request.id}.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"avc1"
    )

    out = cv2.VideoWriter(
        out_path,
        fourcc,
        fps,
        (width, height)
    )

    # ---------------------------------------------------------
    # Load reference image
    # ---------------------------------------------------------

    ref_img = None

    if ref_path and os.path.exists(ref_path):

        ref_img = cv2.imread(
            ref_path
        )

        if ref_img is not None:

            h, w = ref_img.shape[:2]

            max_dim = 1024

            if max(h, w) > max_dim:

                scale = max_dim / max(h, w)

                ref_img = cv2.resize(
                    ref_img,
                    (
                        int(w * scale),
                        int(h * scale)
                    )
                )

            print(
                f"📸 Reference image loaded: "
                f"{ref_img.shape}"
            )

    # ---------------------------------------------------------
    # Fallback detection box
    # ---------------------------------------------------------

    fallback_box = {
        "x1": int(width * 0.4),
        "y1": int(height * 0.4),
        "x2": int(width * 0.6),
        "y2": int(height * 0.6),
        "mask": None
    }

    frame_count = 0

    last_detection = None

    # ---------------------------------------------------------
    # Process frames
    # ---------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # -----------------------------------------------------
        # Detect target directly from prompt
        # -----------------------------------------------------

        detection = detect_object_yolo(
            frame,
            target
        )

        if detection:

            last_detection = detection

        elif (
            frame_count == 30
            and last_detection is None
        ):

            print(
                "⚠️ Using fallback box "
                "(center) because no detection occurred."
            )

            last_detection = fallback_box

        # -----------------------------------------------------
        # Apply operation
        # -----------------------------------------------------

        if last_detection:

            x1 = max(
                0,
                last_detection["x1"] - 5
            )

            y1 = max(
                0,
                last_detection["y1"] - 5
            )

            x2 = min(
                width,
                last_detection["x2"] + 5
            )

            y2 = min(
                height,
                last_detection["y2"] + 5
            )

            mask = last_detection.get(
                "mask"
            )

            # -------------------------------------------------
            # REMOVE
            # -------------------------------------------------

            if operation == "remove":

                if mask is not None:

                    frame = cv2.inpaint(
                        frame,
                        mask * 255,
                        3,
                        cv2.INPAINT_TELEA
                    )

                else:

                    rect_mask = np.zeros(
                        (height, width),
                        dtype=np.uint8
                    )

                    rect_mask[
                        y1:y2,
                        x1:x2
                    ] = 255

                    frame = cv2.inpaint(
                        frame,
                        rect_mask,
                        3,
                        cv2.INPAINT_TELEA
                    )

            # -------------------------------------------------
            # REPLACE
            # -------------------------------------------------

            elif (
                operation == "replace"
                and ref_img is not None
            ):

                roi_width = x2 - x1
                roi_height = y2 - y1

                if (
                    roi_width > 0
                    and roi_height > 0
                ):

                    ref_resized = cv2.resize(
                        ref_img,
                        (
                            roi_width,
                            roi_height
                        )
                    )

                    if mask is not None:

                        roi_mask = mask[
                            y1:y2,
                            x1:x2
                        ]

                        if roi_mask.shape != (
                            roi_height,
                            roi_width
                        ):

                            roi_mask = cv2.resize(
                                roi_mask,
                                (
                                    roi_width,
                                    roi_height
                                )
                            )

                        roi_mask_3ch = np.stack(
                            [roi_mask] * 3,
                            axis=2
                        )

                        roi = frame[
                            y1:y2,
                            x1:x2
                        ]

                        roi[
                            roi_mask_3ch > 0
                        ] = ref_resized[
                            roi_mask_3ch > 0
                        ]

                        frame[
                            y1:y2,
                            x1:x2
                        ] = roi

                    else:

                        frame[
                            y1:y2,
                            x1:x2
                        ] = ref_resized

            # -------------------------------------------------
            # CHANGE TEXT
            # -------------------------------------------------

            elif operation == "change_text":

                if mask is not None:

                    frame = cv2.inpaint(
                        frame,
                        mask * 255,
                        21,
                        cv2.INPAINT_TELEA
                    )

                else:

                    roi = frame[
                        y1:y2,
                        x1:x2
                    ]

                    if roi.size > 0:

                        frame[
                            y1:y2,
                            x1:x2
                        ] = cv2.GaussianBlur(
                            roi,
                            (21, 21),
                            0
                        )

        # -----------------------------------------------------
        # Write frame
        # -----------------------------------------------------

        out.write(frame)

        frame_count += 1

        # -----------------------------------------------------
        # Progress
        # -----------------------------------------------------

        if (
            frame_count % 10 == 0
            and total_frames > 0
        ):

            progress = int(
                (frame_count / total_frames) * 100
            )

            self.update_state(
                state="PROCESSING",
                meta={
                    "progress": progress
                }
            )

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(
        f"✅ Video processing complete: "
        f"{out_path}"
    )

    return {
        "output_url":
            f"/output/output_{self.request.id}.mp4"
    }
