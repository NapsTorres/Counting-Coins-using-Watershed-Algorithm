import cv2
import numpy as np

img = cv2.imread("3sample.jpg")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

sure_bg = cv2.dilate(opening, kernel, iterations=3)

dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
_, sure_fg = cv2.threshold(dist_transform, 0.1*dist_transform.max(), 255, 0)

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

_, markers = cv2.connectedComponents(sure_fg)

markers = markers+1

markers[unknown == 255] = 0

markers = cv2.watershed(img, markers)

colors = np.random.randint(0, 255, (markers.max() + 1, 3))

circle_count = 0

for marker in np.unique(markers):
    if marker == 0:
        continue
    mask = np.zeros(img.shape[:2], dtype="uint8")
    mask[markers == marker] = 255
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True) 
        area = cv2.contourArea(cnt)
        if perimeter == 0:
            circularity = 0
        else:
            circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity > 0.7:
            circle_count += 1
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(img, center, radius, colors[marker].tolist(), 2)
            cv2.putText(img, str(circle_count), (center[0] + radius, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

border_size = 50
border_color = (255, 255, 255)
img_with_border = cv2.copyMakeBorder(img, border_size, 0, 0, 0, cv2.BORDER_CONSTANT, value=border_color)

cv2.putText(img_with_border, "Total Money Count: " + str(circle_count), (10, border_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imshow("Coin Detection", img_with_border)
print("Number of circles detected:", circle_count)
cv2.waitKey(0)
cv2.destroyAllWindows()