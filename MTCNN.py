import cv2
import numpy as np
import os
import shutil
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import logging

class OptimizedFaceMatcher:
    def _init_(self, input_folder, output_folder, reference_face_path, threshold=0.7):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.reference_face_path = reference_face_path
        self.threshold = threshold
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(_name_)
        
        # Initialize FaceNet model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # Initialize face detector
        self.detector_type = self._initialize_detector()
        
        # Create output folders
        os.makedirs(os.path.join(output_folder, "matching_faces"), exist_ok=True)
        os.makedirs(os.path.join(output_folder, "non_matching_faces"), exist_ok=True)

    def _initialize_detector(self):
        """Initialize face detector with fallback"""
        if torch.cuda.is_available():
            try:
                from retinaface import RetinaFace
                self.detector = RetinaFace(quality='normal')
                self.logger.info("Using RetinaFace detector (GPU)")
                return 'retinaface'
            except ImportError:
                self.logger.warning("RetinaFace not installed, falling back to MTCNN")
        
        # Use MTCNN for CPU or fallback (better accuracy than YuNet)
        try:
            self.detector = MTCNN(keep_all=True, device=self.device, post_process=False, min_face_size=60, thresholds=[0.8, 0.9, 0.9])
            self.logger.info("Using MTCNN detector")
            return 'mtcnn'
        except ImportError:
            self.logger.warning("MTCNN not available, attempting YuNet")

        # YuNet as last resort
        script_dir = os.path.dirname(os.path.abspath(_file_))
        model_path = os.path.join(script_dir, "face_detection_yunet_2023mar.onnx")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YuNet model not found at: {model_path}. Download from https://github.com/opencv/opencv_zoo")
        try:
            self.detector = cv2.FaceDetectorYN.create(
                model_path,
                "",
                (320, 320),
                score_threshold=0.8  # Higher threshold for better accuracy
            )
            self.logger.info("Using YuNet detector (CPU)")
            return 'yunet'
        except cv2.error as e:
            self.logger.error(f"Failed to initialize YuNet detector: {str(e)}")
            raise RuntimeError("Could not initialize YuNet detector. Ensure the model file is compatible with your OpenCV version (4.10.0+ recommended).")

    def _align_face(self, image, landmarks):
        """Align face using landmarks (left eye, right eye, nose)"""
        try:
            # Assume landmarks are in format: [[left_eye_x, left_eye_y], [right_eye_x, right_eye_y], [nose_x, nose_y], ...]
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            
            # Compute angle between eyes
            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dY, dX))
            
            # Get center of eyes
            eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
            
            # Get rotation matrix
            M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
            
            # Rotate image
            h, w = image.shape[:2]
            aligned = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
            
            # Adjust landmarks after rotation
            new_landmarks = []
            for lm in landmarks:
                x, y = lm
                new_x = M[0, 0] * x + M[0, 1] * y + M[0, 2]
                new_y = M[1, 0] * x + M[1, 1] * y + M[1, 2]
                new_landmarks.append([new_x, new_y])
            
            return aligned, new_landmarks
        except Exception as e:
            self.logger.warning(f"Failed to align face: {str(e)}")
            return image, landmarks

    def detect_faces(self, image, scale_factor=1.0):
        """Detect faces and return cropped faces, coordinates, and landmarks"""
        try:
            if self.detector_type == 'mtcnn':
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                boxes, _, landmarks = self.detector.detect(image_rgb, landmarks=True)
                if boxes is None:
                    return []
                detected_faces = []
                for box, lm in zip(boxes, landmarks):
                    x1, y1, x2, y2 = map(int, box)
                    face_img = image[y1:y2, x1:x2]
                    if face_img.size > 0:
                        # MTCNN landmarks: [right_eye, left_eye, nose, mouth_left, mouth_right]
                        lm = [[lm[1][0], lm[1][1]], [lm[0][0], lm[0][1]], [lm[2][0], lm[2][1]]]
                        face_img, lm = self._align_face(face_img, lm)
                        orig_x1, orig_y1 = int(x1 / scale_factor), int(y1 / scale_factor)
                        orig_w, orig_h = int((x2 - x1) / scale_factor), int((y2 - y1) / scale_factor)
                        detected_faces.append((face_img, (orig_x1, orig_y1, orig_w, orig_h), lm))
                return detected_faces
            
            elif self.detector_type == 'yunet':
                h, w = image.shape[:2]
                self.detector.setInputSize((w, h))
                _, faces = self.detector.detect(image)
                if faces is None:
                    return []
                detected_faces = []
                for face in faces:
                    x, y, w, h = list(map(int, face[:4]))
                    landmarks = face[4:14].reshape((5, 2))  # YuNet provides 5 landmarks
                    face_img = image[y:y+h, x:x+w]
                    if face_img.size > 0:
                        lm = [landmarks[0], landmarks[1], landmarks[2]]  # left_eye, right_eye, nose
                        face_img, lm = self._align_face(face_img, lm)
                        orig_x, orig_y = int(x / scale_factor), int(y / scale_factor)
                        orig_w, orig_h = int(w / scale_factor), int(h / scale_factor)
                        detected_faces.append((face_img, (orig_x, orig_y, orig_w, orig_h), lm))
                return detected_faces
            
            else:  # retinaface
                faces = self.detector.predict(image)
                detected_faces = []
                for face in faces:
                    x1, y1, x2, y2 = face['x1'], face['y1'], face['x2'], face['y2']
                    landmarks = face['landmarks']  # [[right_eye], [left_eye], [nose], [mouth_right], [mouth_left]]
                    face_img = image[int(y1):int(y2), int(x1):int(x2)]
                    if face_img.size > 0:
                        lm = [landmarks[1], landmarks[0], landmarks[2]]  # left_eye, right_eye, nose
                        face_img, lm = self._align_face(face_img, lm)
                        orig_x1, orig_y1 = int(x1 / scale_factor), int(y1 / scale_factor)
                        orig_w, orig_h = int((x2 - x1) / scale_factor), int((y2 - y1) / scale_factor)
                        detected_faces.append((face_img, (orig_x1, orig_y1, orig_w, orig_h), lm))
                return detected_faces
        except Exception as e:
            self.logger.error(f"Error detecting faces: {str(e)}")
            return []

    def extract_embedding(self, face):
        """Extract FaceNet embedding with enhanced preprocessing"""
        try:
            # Apply histogram equalization for better lighting
            if len(face.shape) == 3:
                face_yuv = cv2.cvtColor(face, cv2.COLOR_BGR2YUV)
                face_yuv[:, :, 0] = cv2.equalizeHist(face_yuv[:, :, 0])
                face = cv2.cvtColor(face_yuv, cv2.COLOR_YUV2BGR)
            
            # Resize with high-quality interpolation
            face = cv2.resize(face, (160, 160), interpolation=cv2.INTER_CUBIC)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor and normalize
            face = torch.tensor(face).permute(2, 0, 1).float()
            face = (face - 127.5) / 128.0
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.resnet(face.unsqueeze(0).to(self.device))
            return embedding.cpu().numpy().flatten()
        except Exception as e:
            self.logger.error(f"Error extracting embedding: {str(e)}")
            return None

    def compare_faces(self, emb1, emb2):
        """Compare faces with cosine similarity"""
        return cosine_similarity([emb1], [emb2])[0][0]

    def process_images(self):
        """Main processing function with optimizations"""
        self.logger.info(f"Loading reference image: {self.reference_face_path}")
        ref_image = cv2.imread(self.reference_face_path)
        if ref_image is None:
            raise ValueError(f"Could not read reference image: {self.reference_face_path}")
        
        ref_faces = self.detect_faces(ref_image)
        if not ref_faces:
            raise ValueError("No face found in reference image!")
        
        ref_embedding = self.extract_embedding(ref_faces[0][0])
        if ref_embedding is None:
            raise ValueError("Failed to extract embedding from reference face")
        self.logger.info("Reference embedding extracted")

        image_files = []
        for root, _, files in os.walk(self.input_folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))

        for image_path in tqdm(image_files, desc="Processing images"):
            try:
                image = cv2.imread(image_path)
                if image is None:
                    self.logger.warning(f"Could not read image: {image_path}")
                    continue

                # Downscale large images for faster detection
                h, w = image.shape[:2]
                scale_factor = 1.0
                if max(h, w) > 2000:
                    scale_factor = 2000 / max(h, w)
                    small_img = cv2.resize(image, (int(w*scale_factor), int(h*scale_factor)))
                else:
                    small_img = image

                faces = self.detect_faces(small_img, scale_factor)
                is_match = False
                best_similarity = 0

                for face, _, _ in faces:
                    embedding = self.extract_embedding(face)
                    if embedding is None:
                        continue
                    similarity = self.compare_faces(ref_embedding, embedding)
                    best_similarity = max(best_similarity, similarity)
                    self.logger.debug(f"Image: {os.path.basename(image_path)}, Similarity: {similarity:.2f}")
                    if similarity > self.threshold:
                        is_match = True
                        break

                output_folder = "matching_faces" if is_match else "non_matching_faces"
                output_name = f"{'match' if is_match else 'nomatch'}{best_similarity:.2f}{os.path.basename(image_path)}"
                output_path = os.path.join(self.output_folder, output_folder, output_name)
                shutil.copy2(image_path, output_path)
                self.logger.info(f"Processed: {os.path.basename(image_path)} - Similarity: {best_similarity:.2f} - {'MATCH' if is_match else 'NO MATCH'}")
            except Exception as e:
                self.logger.error(f"Error processing {image_path}: {str(e)}")

if _name_ == "_main_":
    # Configuration
    INPUT_FOLDER = r"c:\Users\ADMIN\Desktop\23dx43\Imgcf"
    OUTPUT_FOLDER = r"c:\Users\ADMIN\Desktop\23dx43\face_cluster"
    REFERENCE_FACE_PATH = r"c:\Users\ADMIN\Desktop\23dx43\reference_face.jpeg"
    SIMILARITY_THRESHOLD = 0.7

    print("Starting optimized face matching...")
    matcher = OptimizedFaceMatcher(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        reference_face_path=REFERENCE_FACE_PATH,
        threshold=SIMILARITY_THRESHOLD
    )
    matcher.process_images()
    print("Processing complete!")
