from skimage.segmentation import clear_border
import pytesseract
import numpy as np
import imutils
import cv2


class ANPR:
    def __init__(self, minAR=4, maxAR=5, debug=False):
        """

        :param minAR: The minimum aspect ratio used to detect and filter rectangular license plates, which has a default
         value of
        :param maxAR: The maximum aspect ratio of the license plate rectangle, which has a default value of 5
        :param debug: A flag to indicate whether we should display intermediate results in our image processing pipeline
        """
        self.minAR = minAR
        self.maxAR = maxAR
        self.debug = debug

    def debug_imshow(self, title, image, waitKey=False):
        """
        If debug mode is eneabled show the image with supplied tittle
        :param title: The desired OpenCV window title. Window titles should be unique; otherwise OpenCV will replace
        the image in the same-titled window rather than creating a new one.
        :param image: The image to display inside the OpenCV GUI window.
        :param waitKey: A flag to see if the display should wait for a keypress before completing.
        :return:
        """
        if self.debug:
            cv2.imshow(title, image)
            # check to see if we should wait for a keypress
            if waitKey:
                cv2.waitKey(0)

    def locate_license_plate_candidates(self, gray, keep=5):
        """
        Search license plate candidates in the given image, that it's needed to be in gray scale.

        :param gray: This function assumes that the driver script will provide a grayscale image containing a potential
        license plate.
        :param keep: We’ll only return up to this many sorted license plate candidate contours
        :return: list with characters contours.
        """

        # perform a blackhat morphological operation that will allow
        # us to reveal dark regions (i.e., text) on light backgrounds
        # (i.e., the license plate itself)
        rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
        self.debug_imshow("Blackhat", blackhat)

        # find regions in the image that are light
        squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
        light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.debug_imshow("Light Regions", light)

        # compute the Scharr gradient representation of the blackhat
        # image in the x-direction and then scale the result back to
        # the range [0, 255]
        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,
                          dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
        gradX = gradX.astype("uint8")
        self.debug_imshow("Scharr", gradX)

        # blur the gradient representation, applying a closing
        # operation, and threshold the image using Otsu's method
        gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
        thresh = cv2.threshold(gradX, 0, 255,
                               cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.debug_imshow("Grad Thresh", thresh)

        # perform a series of erosions and dilations to clean up the
        # thresholded image
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        self.debug_imshow("Grad Erode/Dilate", thresh)

        # take the bitwise AND between the threshold result and the
        # light regions of the image
        thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)
        self.debug_imshow("Final", thresh, waitKey=True)

        # find contours in the thresholded image and sort them by
        # their size in descending order, keeping only the largest
        # ones
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]
        # return the list of contours
        return cnts

    def locate_license_plate(self, gray, candidates, clearBorder=False):
        """
         Find the most likely contour containing a license plate from our set of candidates

        :param gray: input grayscale image.
        :param candidates: The license plate contour candidates.
        :param clearBorder: A flag indicating whether our pipeline should eliminate any contours that touch the edge
        of the image
        :return: license Plate ROI and contour
        """
        # initialize the license plate contour and ROI
        lp_cnt = None
        roi = None
        # loop over the license plate candidate contours
        for c in candidates:
            # compute the bounding box of the contour and then use
            # the bounding box to derive the aspect ratio
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)

            # check to see if the aspect ratio is rectangular
            if self.minAR <= ar <= self.maxAR:
                # store the license plate contour and extract the
                # license plate from the grayscale image and then
                # threshold it
                lp_cnt = c
                license_plate = gray[y:y + h, x:x + w]
                roi = cv2.threshold(license_plate, 0, 255,
                                    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

                # check to see if we should clear any foreground
                # pixels touching the border of the image
                # (which typically, not but always, indicates noise)
                if clearBorder:
                    roi = clear_border(roi)
                # display any debugging information and then break
                # from the loop early since we have found the license
                # plate region
                self.debug_imshow("License Plate", license_plate)
                self.debug_imshow("ROI", roi, waitKey=True)
                break
                # return a 2-tuple of the license plate ROI and the contour
                # associated with it
            return roi, lp_cnt

    def build_tesseract_options(self, psm=7):
        """
        Use PyTesseract to extract the string of the detected license plate

        :param psm: Tesseract’s setting indicating layout analysis of the document/image. There are 13 modes of
        operation, but we will default to 7 — “treat the image as a single text line” — per the psm parameter default.
        :return: concatenate string with psm and whitelist into a formatted string with these option parameters.
        """
        # tell Tesseract to only OCR alphanumeric characters
        alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        options = "-c tessedit_char_whitelist={}".format(alphanumeric)
        # set the PSM mode
        options += " --psm {}".format(psm)
        # return the built options string
        return options

    def find_and_ocr(self, image, psm=7, clearBorder=False):
        """
        Brings all the components together in one centralized place so our driver script can instantiate a
        ANPR object, and then make a single function call.

        :param image: The three-channel color image of the rear (or front) of a car with a license plate tag.
        :param psm: The Tesseract Page Segmentation Mode.
        :param clearBorder: The flag indicating whether we’d like to clean up contours touching the border of the
        license plate ROI
        :return: 2-tuple of the OCR'd license plate text along with the contour associated with the license plate region
        """
        # initialize the license plate text
        lp_text = None
        # convert the input image to grayscale, locate all candidate
        # license plate regions in the image, and then process the
        # candidates, leaving us with the *actual* license plate
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        candidates = self.locate_license_plate_candidates(gray)
        lp, lp_cnt = self.locate_license_plate(gray, candidates, clearBorder=clearBorder)
        # only OCR the license plate if the license plate ROI is not
        # empty
        if lp is not None:
            # OCR the license plate
            options = self.build_tesseract_options(psm=psm)
            lp_text = pytesseract.image_to_string(lp, config=options)
            self.debug_imshow("License Plate", lp)
        # return a 2-tuple of the OCR'd license plate text along with
        # the contour associated with the license plate region
        return lp_text, lp_cnt
