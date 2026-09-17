import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

FACE_CASCADE = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
EYE_CASCADE = cv2.data.haarcascades + 'haarcascade_eye.xml'
SMILE_CASCADE = cv2.data.haarcascades + 'haarcascade_smile.xml'
SCREENSHOT_DIR = Path('screenshots')


class FacialExpressionAnalyzer:
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE)
        self.eye_cascade = cv2.CascadeClassifier(EYE_CASCADE)
        self.smile_cascade = cv2.CascadeClassifier(SMILE_CASCADE)
    
    def analyze_face(self, face_image):
        eyes = self.eye_cascade.detectMultiScale(face_image, 1.1, 4)
        smiles = self.smile_cascade.detectMultiScale(
            face_image, 
            scaleFactor=1.8,
            minNeighbors=20,
            minSize=(25, 25)
        )
        eye_aspect_ratio = self.calculate_eye_aspect(face_image, eyes)
        emotion, confidence = self.classify_emotion(len(eyes), len(smiles), eye_aspect_ratio)
        
        return emotion, confidence, eyes, smiles
    
    def calculate_eye_aspect(self, image, eyes):
        if len(eyes) < 2:
            return 0
        
        ratios = []
        for (x, y, w, h) in eyes[:2]:
            aspect_ratio = w / h if h != 0 else 0
            ratios.append(aspect_ratio)
        
        return np.mean(ratios) if ratios else 0
    
    def classify_emotion(self, eyes, smiles, eye_aspect):
        if eyes >= 2 and smiles >= 1:
            confidence = min(0.95, 0.7 + (smiles * 0.1))
            return "Happy", confidence
        
        elif eyes >= 2 and eye_aspect > 1.8 and smiles == 0:
            confidence = 0.75
            return "Surprise", confidence
        
        elif eyes >= 2 and smiles == 0 and eye_aspect < 1.2:
            confidence = 0.65
            return "Sad", confidence
        
        elif eyes >= 2 and smiles == 0:
            confidence = 0.8
            return "Neutral", confidence
        
        else:
            confidence = 0.3
            return "Unknown", confidence
    
    def draw_face_analysis(self, frame, face, emotion, confidence, eyes, smiles):
        x, y, w, h = face
        
        colors = {
            "Happy": (0, 255, 0),
            "Sad": (255, 0, 0),
            "Surprise": (0, 255, 255),
            "Neutral": (200, 200, 200),
            "Unknown": (0, 0, 255)
        }
        
        color = colors.get(emotion, (255, 255, 255))
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        
        text = f"{emotion} ({confidence:.1%})"
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
    print("Press 's' to save a screenshot, or 'q' to quit\n")
    
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
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        frame = cv2.flip(frame, 1)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = analyzer.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for face in faces:
            x, y, w, h = face
            face_roi = gray[y:y+h, x:x+w]
            
            emotion, confidence, eyes, smiles = analyzer.analyze_face(face_roi)
            analyzer.draw_face_analysis(frame, face, emotion, confidence, eyes, smiles)
        
        cv2.putText(frame, f"Faces: {len(faces)} | Frame: {frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "S: Screenshot | Q: Quit", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        
        cv2.imshow('Facial Expression Detector', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            screenshot_path = save_screenshot(frame)
            if screenshot_path:
                print(f"Screenshot saved: {screenshot_path}")
            else:
                print("Error: Could not save screenshot")
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Program closed")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram stopped")
    except Exception as e:
        print(f"Error: {e}")

# Facial Expression Detection System - Real-time emotion recognition from webcam using cascade classifiers
