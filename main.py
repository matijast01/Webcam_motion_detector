import cv2
import glob
from emailing import send_email
import os
from threading import Thread
import time

video = cv2.VideoCapture(0)
time.sleep(1)

first_frame = None
status_list = []
count = 1


def clean_folder():
    images = glob.glob("images/*.png")
    for image in images:
        os.remove(image)


while True:
    status = 0
    check, frame = video.read()

    # Grayscale and Gaussian Blur to reduce data size while computing data
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame_gau = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    # Save first frame data
    if first_frame is None:
        first_frame = gray_frame_gau

    # Get the delta frame
    delta_frame = cv2.absdiff(first_frame, gray_frame_gau)

    # Get a frame based on a threshold
    thresh_frame = cv2.threshold(delta_frame, 75, 255, cv2.THRESH_BINARY)[1]

    # dilate the frame
    dil_frame = cv2.dilate(thresh_frame, None, iterations=2)

    cv2.imshow("My Video", dil_frame)

    # Find contours of the objects shown by dil_frame
    contours, check = cv2.findContours(dil_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # check all contours
    for contour in contours:

        # if Contour is small consider it a false object
        if cv2.contourArea(contour) < 5000:
            continue
        # get a bounding rectangle for the contour and draw it on color frame
        x, y, w, h = cv2.boundingRect(contour)
        rectangle = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

        # check if object has entered frame
        if rectangle.any():
            status = 1
            cv2.imwrite(f"images/{count}.png", frame)
            count += 1

    status_list.append(status)
    status_list = status_list[-2:]

    if status_list[0] == 1 and status_list[1] == 0:
        all_images = glob.glob("images/*.png")
        index = int(len(all_images) / 2)
        image_with_object = all_images[index]
        email_thread = Thread(target=send_email, args=(image_with_object, ))
        email_thread.daemon = True

        email_thread.start()

    print(status_list)

    cv2.imshow("Video", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

video.release()
clean_folder()
