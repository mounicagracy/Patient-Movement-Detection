#!/usr/bin/python3.8

"""
This script is meant to detecting the patient's motion thereby alerting the staff incharge.
"""

import threading

import cv2
import pyttsx3


def voice_alarm(sound)->None:
    """Alert the user

    Args:
        sound (obj): it is a sound engine that process the text to speech task.
    """
    try:
        sound.say("patient is moving")
        sound.runAndWait()
    except RuntimeError as e:
        pass

STATUS_LIST = [None, None]

# Initiate the text to speech package; extract the desired voice and set it as
# property for our script.
alarm_sound = pyttsx3.init()    # Initiates the voice object.
voices = alarm_sound.getProperty('voices')  # Getting the available voices.
alarm_sound.setProperty('voice', voices[11].id) # Setting the desired voice
# Setting the desired  voice rate at 175 words per minute.
alarm_sound.setProperty('rate', 175)

# Initialize a video capture object
video = cv2.VideoCapture(0) # Video capturing starts in 1st cam.
INITIAL_FRAME = None    # Initial frame in the video set to None.

while True:
    _, frame = video.read()
    frame = cv2.flip(frame, 1)
    STATUS = 0

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_frame = cv2.GaussianBlur(gray_frame, (13, 13), 0)
    # cv2.imshow("gussian", blur_frame)   # test code

    if INITIAL_FRAME is None:
        INITIAL_FRAME = gray_frame
        continue

    delta_frame = cv2.absdiff(INITIAL_FRAME, blur_frame)
    threshold_frame = cv2.threshold(delta_frame, 50, 255, cv2.THRESH_BINARY)[1]

    (contours, _) = cv2.findContours(threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # DO: note the point of change, if found draw a reactangle over the region
    # of changes.
    for c in contours:

        if cv2.contourArea(c) < 10000:
            continue
        STATUS = STATUS + 1
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
    STATUS_LIST.append(STATUS)  # type: ignore

    # DO: start alerting concurrently  if the STATUS_LIST get updated
    if STATUS_LIST[-1] >= 1 and STATUS_LIST[-2] == 0:  # type: ignore

        alarm = threading.Thread(target=voice_alarm, args=(alarm_sound,))
        alarm.start()
        # alarm.join()

    cv2.imshow('motion detector', frame)    # Display the video
    # DO: exit on pressing q

    if cv2.waitKey(1) == ord('q'):
        break

# DO: close the videp capturing object, destory all the GUI windows and stop
# the alert.
alarm_sound.stop()
video.release()
cv2.destroyAllWindows()