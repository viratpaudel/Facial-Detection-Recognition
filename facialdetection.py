import cv2
import numpy as np
import time
from datetime import datetime
from pathlib import Path
from collections import Counter, deque
import json
import csv

FACE_CASCADE = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
EYE_CASCADE = cv2.data.haarcascades + 'haarcascade_eye.xml'
SMILE_CASCADE = cv2.data.haarcascades + 'haarcascade_smile.xml'
SCREENSHOT_DIR = Path('screenshots')
LOGS_DIR = Path('emotion_logs')
STATS_FILE = LOGS_DIR / 'emotion_stats.json'


class EmotionLogger:
    """Logs emotion data and statistics to files."""

    def __init__(self):
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.csv_file = LOGS_DIR / f'emotions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        self.stats = {
            'happy': 0,
            'sad': 0,
            'angry': 0,
            'surprise': 0,
            'total_frames': 0,
            'session_start': datetime.now().isoformat(),
            'faces_detected': 0
        }
        self._init_csv()

    def _init_csv(self):
        """Initialize CSV file with headers."""
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'emotion', 'confidence', 'faces_count', 'fps'])

    def log_emotion(self, emotion, confidence, faces_count, fps):
        """Log emotion data to CSV file."""
        normalized_emotion = emotion.lower()
        if normalized_emotion not in self.stats:
            return

        timestamp = datetime.now().isoformat()
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, emotion, f"{confidence:.2f}", faces_count, f"{fps:.1f}"])

        self.stats[normalized_emotion] += 1
        self.stats['total_frames'] += 1

    def update_faces_detected(self, count):
        """Update total faces detected."""
        self.stats['faces_detected'] += count

    def save_stats(self):
        """Save statistics to JSON file."""
        self.stats['session_end'] = datetime.now().isoformat()
        with open(STATS_FILE, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"Statistics saved to: {STATS_FILE}")

    def get_stats_summary(self):
        """Get a formatted summary of emotion statistics."""
        if self.stats['total_frames'] == 0:
            return "No data recorded yet"

        summary = "--- Emotion Statistics ---\n"
        for emotion, count in sorted(self.stats.items()):
            if emotion not in ['session_start', 'session_end'] and isinstance(count, int) and count > 0:
                percentage = (count / self.stats['total_frames']) * 100
                summary += f"{emotion.capitalize()}: {count} ({percentage:.1f}%)\n"
        summary += f"Total Frames: {self.stats['total_frames']}\n"
        summary += f"Total Faces Detected: {self.stats['faces_detected']}"
        return summary


class PerformanceMonitor:
    """Monitors performance metrics."""

    def __init__(self, window_size=30):
        self.frame_times = deque(maxlen=window_size)
        self.detection_times = deque(maxlen=window_size)

    def add_frame_time(self, elapsed):
        """Add frame processing time."""
        self.frame_times.append(elapsed)

    def add_detection_time(self, elapsed):
        """Add detection processing time."""
        self.detection_times.append(elapsed)

    def get_avg_frame_time(self):
        """Get average frame processing time in ms."""
        return (np.mean(self.frame_times) * 1000) if self.frame_times else 0

    def get_avg_detection_time(self):
        """Get average detection time in ms."""
        return (np.mean(self.detection_times) * 1000) if self.detection_times else 0


class FacialExpressionAnalyzer:

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE)
        self.eye_cascade = cv2.CascadeClassifier(EYE_CASCADE)
        self.smile_cascade = cv2.CascadeClassifier(SMILE_CASCADE)
        self.emotion_history = deque(maxlen=12)
        self.logger = EmotionLogger()
        self.performance = PerformanceMonitor()
        self.face_tracker = {}
        self.next_face_id = 0

    def analyze_face(self, face_image):
        start_time = time.perf_counter()

        eyes = self.eye_cascade.detectMultiScale(face_image, 1.1, 4)
        smiles = self.smile_cascade.detectMultiScale(
            face_image,
            scaleFactor=1.8,
            minNeighbors=20,
            minSize=(25, 25)
        )
        eye_aspect_ratio = self.calculate_eye_aspect(face_image, eyes)
        emotion, confidence = self.classify_emotion(len(eyes), len(smiles), eye_aspect_ratio)

        detection_time = time.perf_counter() - start_time
        self.performance.add_detection_time(detection_time)

        return emotion, confidence, eyes, smiles

    def calculate_eye_aspect(self, image, eyes):
        if len(eyes) < 2:
            return 0

        ratios = []
        for (x, y, w, h) in eyes[:2]:
            aspect_ratio = w / h if h != 0 else 0
            ratios.append(aspect_ratio)

        return np.mean(ratios) if ratios else 0

    def record_emotion(self, emotion):
        if emotion:
            self.emotion_history.append(emotion)

    def reset_emotion_history(self):
        self.emotion_history.clear()

    def get_dominant_emotion(self):
        if not self.emotion_history:
            return "Happy"
        counts = Counter(self.emotion_history)
        return counts.most_common(1)[0][0]

    def classify_emotion(self, eyes, smiles, eye_aspect):
        if eyes >= 2 and smiles >= 1:
            confidence = min(0.95, 0.7 + (smiles * 0.1))
            return "Happy", confidence

        elif eyes >= 2 and eye_aspect > 1.8 and smiles == 0:
            confidence = 0.75
            return "Surprise", confidence

        elif eyes >= 2 and smiles == 0 and eye_aspect < 1.0:
            confidence = 0.6
            return "Angry", confidence

        elif eyes >= 2 and smiles == 0 and eye_aspect < 1.2:
            confidence = 0.65
            return "Sad", confidence

        else:
            confidence = 0.5
            return "Happy", confidence

    def draw_face_analysis(self, frame, face, emotion, confidence, eyes, smiles, face_id=None):
        x, y, w, h = face

        colors = {
            "Happy": (0, 255, 0),
            "Sad": (255, 0, 0),
            "Angry": (0, 0, 255),
            "Surprise": (0, 255, 255),
        }

        color = colors.get(emotion, (255, 255, 255))

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

        text = f"{emotion} ({confidence:.1%})"
        if face_id is not None:
            text = f"ID:{face_id} | {text}"

        label_top = max(0, y - 35)
        cv2.rectangle(frame, (x, label_top), (x + 300, y), color, -1)
        cv2.putText(frame, text, (x + 5, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        for (ex, ey, ew, eh) in eyes[:2]:
            cv2.rectangle(frame, (x + ex, y + ey),
                         (x + ex + ew, y + ey + eh), (255, 0, 255), 2)

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(frame, (x + sx, y + sy),
                         (x + sx + sw, y + sy + sh), (0, 255, 0), 2)


def save_screenshot(frame):
    """Save the current annotated frame and return its path."""
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    screenshot_path = SCREENSHOT_DIR / f'facial_detection_{timestamp}.jpg'

    if cv2.imwrite(str(screenshot_path), frame):
        return screenshot_path
    return None


def main():
    print("\nStarting facial expression detection...")
    print("Press 's' to save a screenshot, 'r' to reset emotion tracking,")
    print("'p' to show statistics, or 'q' to quit\n")

    analyzer = FacialExpressionAnalyzer()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("Camera opened successfully!")
    print(f"Screenshots will be saved in: {SCREENSHOT_DIR.resolve()}")
    print(f"Emotion logs will be saved in: {LOGS_DIR.resolve()}\n")

    frame_count = 0
    fps = 0.0
    previous_time = time.perf_counter()

    while True:
        frame_start = time.perf_counter()
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = analyzer.face_cascade.detectMultiScale(gray, 1.3, 5)

        if faces:
            detected_emotions = []
            analyzer.logger.update_faces_detected(len(faces))

            for face_idx, face in enumerate(faces):
                x, y, w, h = face
                face_roi = gray[y:y+h, x:x+w]

                emotion, confidence, eyes, smiles = analyzer.analyze_face(face_roi)
                detected_emotions.append(emotion)
                analyzer.draw_face_analysis(frame, face, emotion, confidence, eyes, smiles, face_idx)

            for emotion in detected_emotions:
                analyzer.record_emotion(emotion)
                analyzer.logger.log_emotion(emotion, 0.0, len(faces), fps)
        else:
            analyzer.record_emotion("Happy")
            analyzer.logger.log_emotion("Happy", 0.0, 0, fps)

        current_time = time.perf_counter()
        elapsed = current_time - previous_time
        if elapsed > 0:
            current_fps = 1.0 / elapsed
            fps = current_fps if fps == 0 else (0.9 * fps + 0.1 * current_fps)
        previous_time = current_time

        frame_time = time.perf_counter() - frame_start
        analyzer.performance.add_frame_time(frame_time)

        dominant_emotion = analyzer.get_dominant_emotion()
        cv2.putText(frame, f"Faces: {len(faces)} | Frame: {frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Mood: {dominant_emotion}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        avg_frame_ms = analyzer.performance.get_avg_frame_time()
        avg_detect_ms = analyzer.performance.get_avg_detection_time()
        cv2.putText(frame, f"Frame: {avg_frame_ms:.1f}ms | Detect: {avg_detect_ms:.1f}ms", (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)
        cv2.putText(frame, "S: Screenshot | R: Reset | P: Stats | Q: Quit", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow('Facial Expression Detector', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            screenshot_path = save_screenshot(frame)
            if screenshot_path:
                print(f"Screenshot saved: {screenshot_path}")
            else:
                print("Error: Could not save screenshot")
        elif key == ord('r'):
            analyzer.reset_emotion_history()
            print("Emotion tracking reset.")
        elif key == ord('p'):
            stats_summary = analyzer.logger.get_stats_summary()
            print(f"\n{stats_summary}\n")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    analyzer.logger.save_stats()
    print("Program closed")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram stopped")
    except Exception as e:
        print(f"Error: {e}")

# Facial Expression Detection System - Real-time emotion recognition from webcam using cascade classifiers
