import cv2
import mediapipe as mp
import time
import math

# Pose Detector class definition
class poseDetector():
    def __init__(self):
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=False,
                                     model_complexity=0,
                                     smooth_landmarks=True,
                                     enable_segmentation=False,
                                     smooth_segmentation=True,
                                     min_detection_confidence=0.5,
                                     min_tracking_confidence=0.5)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        return self.lmList

# Face Mesh Setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

my_drawing_specs = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)

# Main function combining pose detection and face mesh
def main():
    cap = cv2.VideoCapture(0)  # Capture from default camera
    pTime = 0
    detector = poseDetector()

    # Create a named window for full screen
    cv2.namedWindow("Combined Pose and Face Mesh", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Combined Pose and Face Mesh", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Create the Mediapipe face mesh object
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

        # Loop to process camera frames
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                print("Failed to capture image")
                break

            img = cv2.flip(img, 1)  # Flip image for mirror effect
            img = cv2.resize(img, (900, 600))  # Resize if needed

            # Pose detection
            detector.findPose(img)
            lmList = detector.getPosition(img)
            
            # Face mesh detection
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(imgRGB)

            # Drawing face landmarks if detected
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=img,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    mp_drawing.draw_landmarks(
                        image=img,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=my_drawing_specs
                    )

            # Calculate FPS
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            # Display FPS on image
            cv2.putText(img, f'FPS: {int(fps)}', (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

            # Display the image in fullscreen
            cv2.imshow("Combined Pose and Face Mesh", img)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
