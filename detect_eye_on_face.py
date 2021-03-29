import cv2

model = 'res10_300x300_ssd_iter_140000.caffemodel'
config = 'deploy.prototxt'
#model = 'opencv_face_detector_uint8.pb'
#config = 'opencv_face_detector.pbtxt'

eye_cascPath = 'haarcascade_eye_tree_eyeglasses.xml'  #eye detect model
eyeCascade = cv2.CascadeClassifier(eye_cascPath)

def check_eye(frame):
    eyes = eyeCascade.detectMultiScale(
    frame,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    # flags = cv2.CV_HAAR_SCALE_IMAGE
    )
    if len(eyes) == 0:
        print('no eyes!!!')
    else:
        print('eyes!!!')
    return eyes
    
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Camera open failed!')
    exit()

net = cv2.dnn.readNet(model, config)

if net.empty():
    print('Net open failed!')
    exit()

while True:
    _, frame = cap.read()
    if frame is None:
        break

    blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104, 177, 123))
    net.setInput(blob)
    detect = net.forward()

    detect = detect[0, 0, :, :]
    (h, w) = frame.shape[:2]

    for i in range(detect.shape[0]):
        confidence = detect[i, 2]
        if confidence < 0.5:
            break

        x1 = int(detect[i, 3] * w)
        y1 = int(detect[i, 4] * h)
        x2 = int(detect[i, 5] * w)
        y2 = int(detect[i, 6] * h)
        
        #사각형그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))
        # 사각형을 그릴 이미지, 사각형의 좌측상단좌표, 우측하단좌표, 테두리 색, 테두리 두께
        #라벨과 라벨붙이기
        label = 'Face: %4.3f' % confidence
        cv2.putText(frame, label, (x1, y1 - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        # 텍스트를 넣을 이미지, 텍스트 내용, 텍스트 시작 좌측하단좌표, 글자체, 글자크기, 글자색, 글자두께, cv2.LINE_AA(좀 더 예쁘게 해주기 위해)
        test = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        a=check_eye(test)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()