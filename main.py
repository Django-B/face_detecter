# import the opencv library
import cv2


def detect_face():
    cascade_path = '/home/django/code/camera_ai/filters/haarcascade_frontalface_default.xml'
    # cascade_path = '/home/django/code/camera_ai/filters/haarcascade_eye.xml'
    # cascade_path = '/home/django/code/camera_ai/filters/haarcascade_smile.xml'
    # cascade_path = '/home/django/code/camera_ai/filters/haarcascade_upperbody.xml'
    # cascade_path = '/home/django/code/camera_ai/filters/haarcascade_frontalcatface.xml'
    clf = cv2.CascadeClassifier(cascade_path)
    
    camera = cv2.VideoCapture(0)

    while True:
        _, frame = camera.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = clf.detectMultiScale(
            gray, # Image
            scaleFactor=1.1, # Коэфицент масштабирования
            minNeighbors=5, # Минимальное число соседей, чем больше, тем строже критерии(найдется меньше лиц) и наоборот, рекомендуемо 5
            minSize=(30, 30), # Минимальный размер объекта
            flags=cv2.CASCADE_SCALE_IMAGE, # Нужно по документации
        )

        rect_color = (255, 255, 0)
        for (x, y, width, height) in faces:
            cv2.rectangle(frame, (x, y), (x+width, y+height), rect_color, 2)
            cv2.putText(frame, 'Detect', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, rect_color, 2)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()

detect_face()
