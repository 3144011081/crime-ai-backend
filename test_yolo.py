import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="Test YOLO and ByteTrack on video")
parser.add_argument("--max_frames", type=int, default=150, help="Max frames to process (0 for full video)")
parser.add_argument("--video", type=str, default="videos/hello1.mp4", help="Path to video file")
args, _ = parser.parse_known_args()

PROJECT_ROOT = Path(__file__).resolve().parent
VIDEO_PATH = Path(args.video) if Path(args.video).is_absolute() else PROJECT_ROOT / args.video

PERSON_MODEL_PATH = PROJECT_ROOT / "models" / "drone_person_detector_best.pt"
WEAPON_MODEL_PATH = PROJECT_ROOT / "models" / "weapon_detector_best.pt"
if not WEAPON_MODEL_PATH.exists() and (PROJECT_ROOT / "models" / "firearmbest.pt").exists():
    WEAPON_MODEL_PATH = PROJECT_ROOT / "models" / "firearmbest.pt"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "yolo_test.mp4"


# ============================================================
# CHECK FILES
# ============================================================

print("Video:", VIDEO_PATH)
print("Video exists:", VIDEO_PATH.exists())

print("Person model:", PERSON_MODEL_PATH)
print("Person model exists:", PERSON_MODEL_PATH.exists())

print("Weapon model:", WEAPON_MODEL_PATH)
print("Weapon model exists:", WEAPON_MODEL_PATH.exists())


# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading person model...")

person_model = YOLO(
    str(PERSON_MODEL_PATH)
)

print("Person model loaded.")


print("\nLoading weapon model...")

if WEAPON_MODEL_PATH.exists():
    weapon_model = YOLO(str(WEAPON_MODEL_PATH))
    print("Weapon model loaded.")
else:
    print(f"Weapon model ({WEAPON_MODEL_PATH.name}) not found on disk, using YOLO fallback.")
    weapon_model = YOLO("yolov8n.pt")


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(
    str(VIDEO_PATH)
)

if not cap.isOpened():

    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )


fps = cap.get(
    cv2.CAP_PROP_FPS
)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

print("\nVideo information:")
print("FPS:", fps)
print("Resolution:", width, "x", height)


# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# PROCESS VIDEO
# ============================================================

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    if args.max_frames > 0 and frame_number >= args.max_frames:
        print(f"Reached max frames limit ({args.max_frames}), stopping test.")
        break

    frame_number += 1


    # ========================================================
    # PERSON + BYTETRACK
    # ========================================================

    person_results = person_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.30,
        verbose=False
    )


    # ========================================================
    # WEAPON DETECTION
    # ========================================================

    weapon_results = weapon_model(
        frame,
        imgsz=960,
        conf=0.15,
        iou=0.45,
        verbose=False
    )


    # ========================================================
    # PERSON BOXES
    # ========================================================

    person_count = 0

    if person_results:

        result = person_results[0]

        if result.boxes is not None:

            boxes = result.boxes

            person_count = len(boxes)

            for i in range(len(boxes)):

                xyxy = (
                    boxes.xyxy[i]
                    .cpu()
                    .numpy()
                )

                x1, y1, x2, y2 = map(
                    int,
                    xyxy
                )

                confidence = float(
                    boxes.conf[i]
                )


                # ByteTrack ID

                track_id = None

                if boxes.id is not None:

                    track_id = int(
                        boxes.id[i]
                    )


                # Draw box

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 180, 0),
                    2
                )


                # Label

                if track_id is not None:

                    label = (
                        f"Person ID {track_id} "
                        f"{confidence * 100:.0f}%"
                    )

                else:

                    label = (
                        f"Person "
                        f"{confidence * 100:.0f}%"
                    )


                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 180, 0),
                    2,
                    cv2.LINE_AA
                )


    # ========================================================
    # WEAPON BOXES
    # ========================================================

    weapon_count = 0

    if weapon_results:

        result = weapon_results[0]

        if result.boxes is not None:

            boxes = result.boxes

            weapon_count = len(boxes)

            for i in range(len(boxes)):

                xyxy = (
                    boxes.xyxy[i]
                    .cpu()
                    .numpy()
                )

                x1, y1, x2, y2 = map(
                    int,
                    xyxy
                )

                confidence = float(
                    boxes.conf[i]
                )

                class_id = int(
                    boxes.cls[i]
                )

                weapon_name = (
                    weapon_model.names[
                        class_id
                    ]
                )


                # Draw weapon box

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    3
                )


                label = (
                    f"! {weapon_name} "
                    f"{confidence * 100:.0f}% (+25% Crime Boost)"
                )


                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )


    # ========================================================
    # INFORMATION PANEL
    # ========================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (380, 105),
        (15, 15, 15),
        -1
    )


    cv2.putText(
        frame,
        "YOLO + BYTETRACK TEST",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"People: {person_count}",
        (20, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 180, 0),
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"Weapons: {weapon_count}",
        (20, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # WRITE
    # ========================================================

    writer.write(
        frame
    )


    # ========================================================
    # PROGRESS
    # ========================================================

    if frame_number % 30 == 0:

        print(
            f"Processed frame {frame_number}"
        )


# ============================================================
# RELEASE
# ============================================================

cap.release()
writer.release()


print("\n========================================")
print("YOLO TEST COMPLETE")
print("========================================")

print(
    "Output:",
    OUTPUT_VIDEO
)
