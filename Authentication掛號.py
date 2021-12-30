import numpy as np
import os
import cv2
import sqlite3
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from numpy import expand_dims
from sklearn.svm import SVC
import pickle
from mtcnn.mtcnn import MTCNN
from PIL import Image as Img, ImageTk
import json
import tkinter as tk
import tkinter.messagebox
from DBMS import *
import qrcode
import serial
import time

from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from load_model.keras_loader import load_keras_model, keras_inference


class Authentication:
    def __init__(self, root):
        self.root = root
        self.root.title('登入介面')

        tk.Label(self.root, text="(1)掛號系統", font=(
            '標楷體', 100, "bold", 'underline')).place(x=580, y=50)
        self.frame = tk.LabelFrame(self.root, text='Login', font=("Times", 36))
        self.frame.place(x=810, y=400)

        tk.Label(self.frame, text=' ID Num. ', font=("Times", 24)).grid(
            row=2, column=1, sticky=tk.W)
        self.username = tk.Entry(self.frame)
        self.username.grid(row=2, column=2)
        tk.Label(self.frame, text=' Password ', font=("Times", 24)).grid(
            row=5, column=1, sticky=tk.W)
        self.password = tk.Entry(self.frame, show='*')
        self.password.grid(row=5, column=2)

        # Button
        tk.Button(self.frame, text='Sign In', command=self.login_user, font=("Times", 24)).grid(
            row=8, column=2)

        self.message = tk.Label(text='', fg='Red', font=("Times", 48))
        self.message.place(relx=0.35, rely=0.75)

        def tick():
            d = datetime.datetime.now()
            today = '{:%B %d,%Y}'.format(d)

            mytime = time.strftime('%I:%M:%S%p')
            self.lblInfo.config(text=(mytime + '    ' + today))
            self.lblInfo.after(200, tick)

        self.lblInfo = Label(font=('arial', 32, 'bold'), fg='Dark green')
        self.lblInfo.place(relx=0.35, rely=0.9)
        tick()

    def login_user(self):

        if self.username.get().rstrip() == 'admin' and self.password.get().rstrip() == 'admin':
            root.destroy()
            newroot = tk.Tk()
            application = Hospital_Portal(newroot)
            newroot.mainloop()

        else:
            username = self.username.get()
            if self.username.get().rstrip() == '111':
                username = "J108455441"
            if self.username.get().rstrip() == '222':
                username = "J151303762"
            if self.username.get().rstrip() == '333':
                username = "K260956982"
            with sqlite3.connect(db_name) as conn:
                cursor = conn.cursor()
            finduser = (
                "SELECT * FROM Userlist WHERE IDnumber=? AND Password=?")
            cursor.execute(
                finduser, [(username), (self.password.get())])
            result = cursor.fetchall()

            if result:
                self.frame.place_forget()
                self.camframe = tk.LabelFrame(
                    self.root, text='FaceID', font=("Times", 36))
                self.camframe.place(x=700, y=230)

                self.message['text'] = 'Please look at the webcam'
                root.update()
                with open(json_path, 'r', encoding="utf-8") as f:
                    namelist = json.load(f)
                    id_list = []
                    name_list = []

                    for num in namelist.values():
                        id_list += [str(num['id'])]
                        name_list += [str(num['name'])]
                    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

                    imagesize = (160, 160)
                    in_encoder = Normalizer(norm='l2')

                    coutname = np.zeros(len(os.listdir(DataPath)))
                    self.coutname = coutname
                    while (True):
                        if cv2.waitKey(20) & 0xFF == ord('q'):
                            break
                        ret, frame = cap.read()
                        results = detector.detect_faces(frame)
                        if len(results) > 0:
                            x, y, w, h = results[0]['box']
                            face = frame[y:y+h, x:x+w]
                            resizeimg = cv2.resize(face, imagesize)
                            embedding = self.get_embedding(model, resizeimg)
                            embedding = np.array(embedding).reshape(1, -1)
                            newtestX = in_encoder.transform(embedding)
                            out_encoder = LabelEncoder()
                            out_encoder.fit(trainy)
                            yhat_test = models.predict(newtestX)
                            color = (255, 0, 0)
                            stroke = 2
                            cv2.rectangle(
                                frame, (x, y), (x+w, y+h), color, stroke)
                            yt = yhat_test[0]
                            name_id = namelist[yt]['id']
                            # idnum = namelist[yt]['name']
                            self.coutname[name_id] += 1
                        else:
                            class_id = self.mask(frame)
                            if not class_id:
                                self.message['text'] = '請脫下口罩'
                            else:
                                self.message['text'] = '偵測不到人瞼'
                            root.update()
                        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                        current_image = Img.fromarray(cv2image)
                        frameC = ImageTk.PhotoImage(image=current_image)
                        self.Faceimg = tk.Label(
                            self.camframe, image=frameC)
                        self.Faceimg.grid(row=1, column=1)
                        root.update()
                        allframe = sum(self.coutname)

                        if allframe > 20:
                            cap.release()
                            faceidid = name_list[int(
                                np.argwhere(coutname == max(coutname))[0])]
                            if faceidid == username:
                                self.camframe.place_forget()
                                self.message['text'] = 'Finish Certification!\n你是 {}'.format(
                                    faceidid)
                                cv2.imwrite('./Face/' + username + '/' +
                                            username + '.png', face)
                                root.update()
                                info_qurey = (
                                    "SELECT Event,Medic,IDnumber,Password,Birth,Age,Gender FROM Userlist WHERE IDnumber=? AND Password=?")
                                cursor.execute(
                                    info_qurey, [(username), (self.password.get())])
                                correct = cursor.fetchall()
                                if correct:
                                    for row in correct:
                                        if row[1] == 0:
                                            qrinfo = "事項：" + \
                                                row[0]+"\nID："+row[2]+"\nPW："+row[3]+"\nBirth：" + \
                                                row[4]+"\nAge："+str(row[5]) + \
                                                "\nGender："+str(row[6])
                                            qr = qrcode.make(qrinfo)
                                            qr.save('test.png')
                                            self.qrpng = PhotoImage(
                                                file='test.png')
                                            self.qrframe = tk.LabelFrame(
                                                self.root, text='QRcode', font=("Times", 52))
                                            self.qrframe.place(x=700, y=200)
                                            self.qrimg = tk.Label(
                                                self.qrframe, image=self.qrpng)
                                            self.qrimg.grid(row=1, column=1)
                                            self.message['text'] = "請拍下QR Code用作排隊之用" + \
                                                "\n"+row[0]
                                            self.wait(10)
                                            self.qrframe.place_forget()
                                            application = Authentication(root)
                                        else:
                                            # arduinoData.write(b'%d' % (row[1]))
                                            # arduinoData.write(b'0')
                                            self.message['text'] = "請領藥"
                                            self.wait(5)
                                            application = Authentication(root)

                                        self.username.delete(0, 'end')
                                        self.password.delete(0, 'end')
                                        self.message['text'] = ''
                                        root.update()
                                    break
                            else:
                                self.message['text'] = '驗證失敗，請重新登入!\n你不是{}'.format(
                                    username)
                                self.camframe.place_forget()
                                self.frame.place(x=810, y=400)
                                root.update()
                                break
                    cap.release()
            else:
                self.message['text'] = 'ID Number or Password incorrect. Try again!'
                root.update()

    def wait(self, TimeCountdown):
        while (TimeCountdown > 0):
            time.sleep(1)
            root.update()
            TimeCountdown -= 1

    def get_embedding(self, model, face_pixels):
        face_pixels = face_pixels.astype('float32')
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        samples = expand_dims(face_pixels, axis=0)
        yhat = model.predict(samples)
        return yhat[0]

    def mask(self, image):
        class_id = 1
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
        anchor_sizes = [[0.04, 0.056], [0.08, 0.11],
                        [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
        anchor_ratios = [[1, 0.62, 0.42]] * 5
        anchors = generate_anchors(
            feature_map_sizes, anchor_sizes, anchor_ratios)
        anchors_exp = np.expand_dims(anchors, axis=0)

        height, width, _ = image.shape
        target_shape = (260, 260)
        image_resized = cv2.resize(image, target_shape)
        image_np = image_resized / 255.0
        image_exp = np.expand_dims(image_np, axis=0)
        y_bboxes_output, y_cls_output = keras_inference(Mmodel, image_exp)
        y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
        y_cls = y_cls_output[0]
        bbox_max_scores = np.max(y_cls, axis=1)
        bbox_max_score_classes = np.argmax(y_cls, axis=1)
        keep_idxs = single_class_non_max_suppression(
            y_bboxes, bbox_max_scores, conf_thresh=0.5, iou_thresh=0.4,)
        for idx in keep_idxs:
            class_id = bbox_max_score_classes[idx]
        return class_id


############################################################
if __name__ == '__main__':
    # arduinoData = serial.Serial("COM3", 9600)
    DataPath = 'Face'
    trainy = []
    for file in os.listdir(DataPath):
        trainy.append(file)
    detector = MTCNN()
    model = load_model('facenet_keras.h5')
    Mmodel = load_keras_model(
        'models/face_mask_detection.json', 'models/face_mask_detection.hdf5')
    with open('SVC.pickle', 'rb') as f:
        models = pickle.load(f)
    json_path = 'name.json'
    db_name = 'Database.db'
    root = tk.Tk()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    root.geometry("%dx%d" % (w, h))
    root.state("zoomed")
    application = Authentication(root)

    root.mainloop()
############################################################
