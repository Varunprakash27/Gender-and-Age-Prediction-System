import cv2 as cv
from flask import Flask,render_template,Response,request

page=Flask(__name__)


def faceBox(faceNet,frame):
    frameHeight=frame.shape[0]
    frameWidth=frame.shape[1]
    blob=cv.dnn.blobFromImage(frame, 1.0, (300,300), [104,117,123], swapRB=False)
    faceNet.setInput(blob)
    detection=faceNet.forward()
    bboxs=[]
    for i in range(detection.shape[2]):
        confidence=detection[0,0,i,2]
        if confidence>0.7:
            x1=int(detection[0,0,i,3]*frameWidth)
            y1=int(detection[0,0,i,4]*frameHeight)
            x2=int(detection[0,0,i,5]*frameWidth)
            y2=int(detection[0,0,i,6]*frameHeight)
            bboxs.append([x1,y1,x2,y2])
            cv.rectangle(frame, (x1,y1),(x2,y2),(0,255,0), 1)
    return frame, bboxs


faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"



faceNet=cv.dnn.readNet(faceModel, faceProto)
ageNet=cv.dnn.readNet(ageModel,ageProto)
genderNet=cv.dnn.readNet(genderModel,genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']



padding=20

@page.route('/')
def home():
    return render_template('main.html')


@page.route('/camera')
def camera():
        video=cv.VideoCapture(0)
        while True:
            ret,frame=video.read()
            frame,bboxs=faceBox(faceNet,frame)
            for bbox in bboxs:
                face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
                blob=cv.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPred=genderNet.forward()
                gender=genderList[genderPred[0].argmax()]


                ageNet.setInput(blob)
                agePred=ageNet.forward()
                age=ageList[agePred[0].argmax()]


                label="{},{}".format(gender,age)
                cv.rectangle(frame,(bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0,255,0),-1) 
                cv.putText(frame, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2,cv.LINE_AA)
                ret,buffer=cv.imencode('.jpg',frame)
                frame=buffer.tobytes()
                yield(b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@page.route('/video')
def video():
    return Response(camera(),mimetype='multipart/x-mixed-replace; boundary=frame')

@page.route('/process',methods=['POST','GET'])
def process():
    if request.method=='POST':
        return render_template('face.html')
    else:
        return render_template('error.html')

if __name__=="__main__":
    page.run(debug=True)