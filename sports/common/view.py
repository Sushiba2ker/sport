from typing import Tuple
import cv2
import numpy as np
import numpy.typing as npt
from collections import deque


class ViewTransformer:
    def __init__(
            self,
            source: npt.NDArray[np.float32],
            target: npt.NDArray[np.float32],
            method: str = 'ransac'
    ) -> None:
        """
        Initialize the ViewTransformer with source and target points.

        Args:
            source (npt.NDArray[np.float32]): Source points for homography calculation.
            target (npt.NDArray[np.float32]): Target points for homography calculation.
            method (str): Method for homography calculation.

        Raises:
            ValueError: If source and target do not have the same shape or if they are
                not 2D coordinates.
        """
        if source.shape != target.shape:
            raise ValueError("Source and target must have the same shape.")
        if source.shape[1] != 2:
            raise ValueError("Source and target points must be 2D coordinates.")

        source = source.astype(np.float32)
        target = target.astype(np.float32)
        
        if method == 'ransac':
            self.m, mask = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
        else:
            self.m, _ = cv2.findHomography(source, target)
            
        if not self._validate_homography():
            raise ValueError("Invalid homography matrix detected")
            
    def _validate_homography(self) -> bool:
        """
        Kiểm tra tính hợp lệ của ma trận homography
        """
        if self.m is None:
            return False
            
        det = np.linalg.det(self.m[:2, :2])
        if abs(det) < 1e-6:
            return False
            
        scale_x = np.sqrt(self.m[0,0]**2 + self.m[0,1]**2)
        scale_y = np.sqrt(self.m[1,0]**2 + self.m[1,1]**2)
        if scale_x < 0.1 or scale_x > 10 or scale_y < 0.1 or scale_y > 10:
            return False
            
        return True

    def transform_points(
            self,
            points: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Transform the given points using the homography matrix.

        Args:
            points (npt.NDArray[np.float32]): Points to be transformed.

        Returns:
            npt.NDArray[np.float32]: Transformed points.

        Raises:
            ValueError: If points are not 2D coordinates.
        """
        if points.size == 0:
            return points

        if points.shape[1] != 2:
            raise ValueError("Points must be 2D coordinates.")

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2).astype(np.float32)

    def transform_image(
            self,
            image: npt.NDArray[np.uint8],
            resolution_wh: Tuple[int, int]
    ) -> npt.NDArray[np.uint8]:
        """
        Transform the given image using the homography matrix.

        Args:
            image (npt.NDArray[np.uint8]): Image to be transformed.
            resolution_wh (Tuple[int, int]): Width and height of the output image.

        Returns:
            npt.NDArray[np.uint8]: Transformed image.

        Raises:
            ValueError: If the image is not either grayscale or color.
        """
        if len(image.shape) not in {2, 3}:
            raise ValueError("Image must be either grayscale or color.")
        return cv2.warpPerspective(image, self.m, resolution_wh)

class CameraStabilizer:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(8, 4)  # 8 states, 4 measurements
        
        # Initialize Kalman filter matrices
        self.kalman.measurementMatrix = np.array([
            [1,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0]
        ], np.float32)
        
        self.kalman.transitionMatrix = np.eye(8, dtype=np.float32)
        self.kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        
    def update(self, homography: np.ndarray) -> np.ndarray:
        """
        Ổn định homography matrix sử dụng Kalman filter
        """
        # Extract parameters from homography
        params = self._decompose_homography(homography)
        
        # Update Kalman filter
        measured = np.array(params, np.float32)
        self.kalman.correct(measured)
        predicted = self.kalman.predict()
        
        # Reconstruct homography
        return self._reconstruct_homography(predicted)
        
    def _decompose_homography(self, H: np.ndarray) -> List[float]:
        """
        Phân tách homography thành các thành phần: translation, rotation, scale
        """
        # Implementation here
        pass
        
    def _reconstruct_homography(self, params: np.ndarray) -> np.ndarray:
        """
        Tái tạo homography từ các thành phần
        """
        # Implementation here
        pass

class TemporalFilter:
    def __init__(self, buffer_size: int = 5):
        self.buffer = deque(maxlen=buffer_size)
        
    def smooth(self, homography: np.ndarray) -> np.ndarray:
        """
        Làm mượt homography matrix qua thời gian
        """
        self.buffer.append(homography)
        
        if len(self.buffer) < 2:
            return homography
            
        # Tính weighted average của các homography matrices
        result = np.zeros_like(homography)
        weights = np.linspace(0.5, 1.0, len(self.buffer))
        weights = weights / np.sum(weights)
        
        for i, H in enumerate(self.buffer):
            result += weights[i] * H
            
        return result

class EnhancedViewTransformer(ViewTransformer):
    def __init__(
            self,
            source: npt.NDArray[np.float32],
            target: npt.NDArray[np.float32],
            use_stabilizer: bool = True,
            use_temporal: bool = True
    ) -> None:
        super().__init__(source, target, method='ransac')
        
        self.stabilizer = CameraStabilizer() if use_stabilizer else None
        self.temporal_filter = TemporalFilter() if use_temporal else None
        
    def transform_image(
            self,
            image: npt.NDArray[np.uint8],
            resolution_wh: Tuple[int, int]
    ) -> npt.NDArray[np.uint8]:
        """
        Transform image với các cải tiến stability
        """
        H = self.m.copy()
        
        # Apply stabilization if enabled
        if self.stabilizer:
            H = self.stabilizer.update(H)
            
        # Apply temporal smoothing if enabled
        if self.temporal_filter:
            H = self.temporal_filter.smooth(H)
            
        return cv2.warpPerspective(image, H, resolution_wh)
