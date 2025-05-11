# test_cv_gui.py
import cv2
print(f"OpenCV version: {cv2.__version__}")
# Create a dummy black image
img = cv2.imread("non_existent_image.jpg") # This will be None, which is fine for this test
if img is None:
    import numpy as np
    img = np.zeros((512, 512, 3), dtype=np.uint8)

try:
    cv2.imshow("Test Window", img)
    print("cv2.imshow executed. If you see a black window, GUI is working.")
    cv2.waitKey(0) # Wait for any key press
    cv2.destroyAllWindows()
    print("Test finished.")
except cv2.error as e:
    print(f"OpenCV GUI Error during test: {e}")
    print("This indicates OpenCV still cannot create display windows.")