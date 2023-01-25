import torch
from flask import Flask, render_template, Response
import cv2
app=Flask(__name__)
camera = cv2.VideoCapture(0)

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom

def gen_frames():  
    while True:
        success,frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            #detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            #faces=detector.detectMultiScale(frame,1.1,7)
            #Inference
            results = model(frame)
            print(results)

            labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
            n = len(labels)
            x_shape, y_shape = frame.shape[1], frame.shape[0]
            for i in range(n):
                row = cord[i]
                if row[4] >= 0.2:
                    x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                    bgr = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
#                     cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                                
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)