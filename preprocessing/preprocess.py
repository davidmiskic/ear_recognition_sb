import cv2

class Preprocess:

    def histogram_equlization_rgb(self, img):
        for channel in range(0, img.shape[2],2):
            img[:, :, channel] = cv2.equalizeHist(img[:, :, channel])
        return img

    # Add your own preprocessing techniques here.

    def histogram_equlization_gray(self, img):
        try:img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except:pass
        return cv2.equalizeHist(img)

    def adaptive_thresholding(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(img, (9, 7), 0)
        p_image = cv2.adaptiveThreshold(blur, 60, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
        return cv2.add(img, p_image)

    def adaptive_tresholding_rb(self, img):
        blur = cv2.GaussianBlur(img, (9, 7), 0)
        for channel in range(0,img.shape[2],2):
            r, p_image = cv2.threshold(blur[:, :, channel], 0,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C+ cv2.THRESH_BINARY)
            img[:, :, channel] = p_image
        return img

    def otsu_thresholding(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(img, (1, 1), 0)
        r, p_image = cv2.threshold(blur,0,255,cv2.THRESH_OTSU)
        return p_image

    def canny(self, img):
        try: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except: pass
        edges = cv2.Canny(img, 700, 800)
        return cv2.add(edges, img)