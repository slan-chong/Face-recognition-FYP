import numpy as np
import os
import cv2
from keras.models import load_model
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle
from mtcnn.mtcnn import MTCNN
from PIL import Image, ImageTk
import tkinter as tk
import json


class Page(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)

    def show(self):
        self.lift()


class Page1(Page):
    def __init__(self, *args, **kwargs):
        Page.__init__(self, *args, **kwargs, bg="#36393f")
        self.flag_reset = flag_reset
        self.flag_limframe = flag_limframe

        cam_btn = tk.Button(self, command=self.cam_switch,
                            image=off_img, bg="#36393f")
        cam_btn_lb = tk.Label(self, text="Webcam開關🎬🎬", bg="#36393f",
                              fg="#ffffff", font=("Times", 16))
        limframe_btn = tk.Button(self, command=self.limframe_switch,
                                 image=off_img, bg="#36393f")
        limframe_btn_lb = tk.Label(self, text="30frame", bg="#36393f",
                                   fg="#ffffff", font=("Times", 16))

        cam_img = tk.Label(self, image=None, bg="#36393f")
        acc_lb = tk.Label(self, text="計算正確率\n輸入名字",
                          bg="#36393f", fg="#ffffff", font=("Times", 18))
        acc_text = tk.Label(self, text=None, bg="#36393f",
                            fg="#ffffff", font=("Times", 16))
        reset_bt = tk.Button(self, text="Reset↺↺↺",
                             command=self.reset, bg="#36393f", fg="#ffffff", font=("Times", 16))
        self.acc_entry = tk.Entry(self)
        self.cam_btn = cam_btn
        self.limframe_btn = limframe_btn
        self.cam_img = cam_img
        self.acc_text = acc_text

        acc_lb.pack(anchor=tk.NW)
        self.acc_entry.pack(anchor=tk.NW)
        acc_text.pack(anchor=tk.NW)
        reset_bt.pack(anchor=tk.NW)
        cam_btn_lb.pack()
        cam_btn.pack()
        limframe_btn_lb.pack()
        limframe_btn.pack()
        cam_img.pack()

    def cam_switch(self):
        global flag_cam
        flag_cam = not flag_cam
        if flag_cam:
            self.cam_btn['image'] = on_img
            self.MTCNN()
        else:
            self.cam_btn['image'] = off_img
            self.cam_img['image'] = None
        root.update()

    def reset(self):
        self.flag_reset = True

    def limframe_switch(self):
        self.flag_limframe = not self.flag_limframe
        if self.flag_limframe:
            self.limframe_btn['image'] = on_img
        else:
            self.limframe_btn['image'] = off_img
        root.update()

    def get_embedding(self, model, face_pixels):
        face_pixels = face_pixels.astype('float32')
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        samples = expand_dims(face_pixels, axis=0)
        yhat = model.predict(samples)
        return yhat[0]
############################################################

    def MTCNN(self, *args, **kwargs):

        with open(json_path, 'r', encoding="utf-8") as f:
            namelist = json.load(f)
        id_list = []
        name_list = []
        for num in namelist.values():
            id_list += [str(num['id'])]
            name_list += [str(num['name'])]
        face_cascade = cv2.CascadeClassifier(
            'D:/Python/path/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap = cap
        self.cv2 = cv2
        trainy = []
        for file in os.listdir('Face'):
            trainy.append(file)
        cap = cv2.VideoCapture(0)
        detector = MTCNN()
        with open('SVC.pickle', 'rb') as f:
            models = pickle.load(f)
        imagesize = (160, 160)
        in_encoder = Normalizer(norm='l2')

        coutname = np.zeros(len(os.listdir(DataPath)))
        self.coutname = coutname

        while (flag_cam):
            if self.flag_reset:
                self.coutname = np.zeros(len(os.listdir(DataPath)))
                self.flag_reset = not self.flag_reset
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
            ret, frame = cap.read()
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = detector.detect_faces(frame)
            if len(results) > 0:
                x, y, w, h = results[0]['box']
                face = frame[y:y+h, x:x+w]
                resizeimg = cv2.resize(
                    face, imagesize, interpolation=cv2.INTER_CUBIC)
                embedding = self.get_embedding(model, resizeimg)
                embedding = np.array(embedding).reshape(1, -1)
                newtestX = in_encoder.transform(embedding)
                out_encoder = LabelEncoder()
                out_encoder.fit(trainy)
                yhat_test = models.predict(newtestX)

                color = (255, 0, 0)
                stroke = 2
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, stroke)
                yt = yhat_test[0]
                name_id = namelist[yt]['id']
                name = namelist[yt]['name']
                font = cv2.FONT_HERSHEY_SIMPLEX
                color = (0, 0, 255)
                stroke = 2
                cv2.putText(frame, name, (x, y), font, 2,
                            color, stroke, cv2.LINE_AA)
                self.coutname[name_id] += 1
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            current_image = Image.fromarray(cv2image)
            frame = ImageTk.PhotoImage(image=current_image)
            self.cam_img['image'] = frame
            allframe = sum(self.coutname)
            if self.flag_limframe and allframe > 50:
                self.cam_switch()
                break
############################################################
# 運算Acc
            desiredLabel = self.acc_entry.get()  # 取得名字

            if desiredLabel in name_list:
                Gtrue = self.coutname[name_list.index(desiredLabel)]
                accuracy = (Gtrue / allframe)*100
                self.acc_text['text'] = "Accuracy：" + "%.3f" % accuracy + "%"+"\nPredict Label :" + \
                    str(self.coutname)+"\n總圖數：" + str(allframe)
            else:
                self.acc_text['text'] = "Do not exist"
############################################################
            root.update()
        cap.release()
        cv2.destroyAllWindows()


class Page2(Page):
    def __init__(self, *args, **kwargs):
        Page.__init__(self, *args, **kwargs, bg="#36393f")

        # addimg
        add_lb1 = tk.Label(self, text="新増圖像", bg="#36393f",
                           fg="#ffffff", font=("Times", 12))
        add_lb2 = tk.Label(self, text="請輸入名字", bg="#36393f",
                           fg="#ffffff", font=("Times", 12))
        train_bt = tk.Button(self, text="重新訓練", command=self.train,
                             bg="#36393f", fg="#ffffff", font=("Times", 16))
        add_ok_bt = tk.Button(
            self, text="開始拍照", command=self.addimg, bg="#36393f", fg="#ffffff", font=("Times", 16))
        showimg_lb = tk.Label(self, image=None, bg="#36393f")
        self.add_entry = tk.Entry(self)

        self.showimg_lb = showimg_lb

        add_lb1.grid(row=0, column=1)
        add_lb2.grid(row=1, column=1)
        self.add_entry.grid(row=1, column=2)
        train_bt.grid(row=0, column=4, )
        add_ok_bt.grid(row=1, column=3)
        showimg_lb.grid(row=2, column=1)
############################################################
# 訓練

    def train(self):
        detector = MTCNN()
        self.detector = detector
        model = load_model('facenet_keras.h5')
        imagesize = (160, 160)
        in_encoder = Normalizer(norm='l2')

        trainx, trainy = self.load_folder(DataPath)
        newtrainx = []
        for face_pixels in trainx:
            embedding = self.get_embedding(model, face_pixels)
            newtrainx.append(embedding)
        newtrainx = np.array(newtrainx)
        newtrainX = in_encoder.transform(newtrainx)
        out_encoder = LabelEncoder()
        out_encoder.fit(trainy)
        trainY = out_encoder.transform(trainy)
        models = SVC(kernel='linear', probability=True)
        models.fit(newtrainX, trainy)
        with open('SVC.pickle', 'wb') as f:
            pickle.dump(models, f)

    def get_embedding(self, model, face_pixels):
        face_pixels = face_pixels.astype('float32')
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        samples = expand_dims(face_pixels, axis=0)
        yhat = model.predict(samples)
        return yhat[0]

    def extract_face(self, filename, imagesize=(160, 160)):
        faces = []
        image = cv2.imread(filename)
        results = self.detector.detect_faces(image)
        if len(results) > 0:
            x, y, w, h = results[0]['box']
            if y > 0:
                face = image[y:y+h, x:x+w]
                resizeimg = cv2.resize(
                    face, imagesize, interpolation=cv2.INTER_CUBIC)
                faces.extend([resizeimg])
            return faces
        else:
            return faces

    def load_dataset(self, folder):
        faces = []
        for file in os.listdir(folder):
            path = folder + '/' + file
            face = self.extract_face(path)
            if len(face) > 0:
                faces.extend(face)
        return faces

    def load_folder(self, folder):
        x, y = [], []
        for file in os.listdir(folder):
            path = folder + '/' + file
            faces = self.load_dataset(path)
            labels = [file for _ in range(len(faces))]
            x.extend(faces)
            y.extend(labels)
        return np.array(x), np.array(y)
############################################################
# 新增圖片&.json file

    def getname(self, *args, **kwargs):
        return self.add_entry.get()

    def addimg(self, *args, **kwargs):
        name = self.getname()  # Slan
        coutData = "%03d" % len(os.listdir(DataPath)) + "_"  # 015_
        coutFloderName = coutData + name  # 015_Slan
        coutDataPath = DataPath + '/' + coutFloderName  # Face/015_Slan

        k = 0
        DataPath_list = os.listdir(DataPath)  # {000_A ... 014_Someone}
        for f in DataPath_list:
            if name == DataPath_list[k][4:]:  # Slan == (005_)Slan
                existsData = DataPath_list[k]  # existsData = 005_Slan
                coutDataPath = DataPath + '/' + existsData  # DataPathName = Face/005_Slan
            k += 1

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        self.coutFloderName = coutFloderName
        if not os.path.exists(coutDataPath):
            os.mkdir(coutDataPath)
            ret, frame = cap.read()
            cv2.imwrite(coutDataPath + '/' + name + '_1' + '.png', frame)
            cv2.waitKey()
            cap.release()
            cv2.destroyAllWindows()
            countfolder = len(os.listdir(DataPath))-1
            self.countfolder = countfolder
            self.write_json(countfolder, name)

        else:
            print("This name already exists.")
            ret, frame = cap.read()
            counter = len(os.listdir(coutDataPath))
            counter += 1
            counter = str(counter)
            cv2.imwrite(coutDataPath + '/' + name +
                        '_'+counter + '.png', frame)
            cap.release()
            cv2.destroyAllWindows()

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        current_image = Image.fromarray(cv2image)
        frame = ImageTk.PhotoImage(image=current_image)
        self.frame = frame
        self.showimg_lb['image'] = self.frame

############################################################
# .json function
    def write_json(self, *args, **kwargs):
        name = self.getname()
        json_file = open(json_path, "r", encoding="utf-8")
        namelist = json.load(json_file)
        newname = {self.coutFloderName: {"id": self.countfolder, "name": name}}
        namelist.update(newname)
        with open(json_path, 'w', encoding="utf-8") as f:
            json.dump(namelist, f)

    def del_json(self):
        json_file = open(json_path, "r", encoding="utf-8")
        namelist = json.load(json_file)
        name = self.getname()
        if name in namelist:
            del namelist[name]
        with open(json_path, 'w', encoding="utf-8") as f:
            json.dump(namelist, f)


############################################################

class Page3(Page):
    def __init__(self, *args, **kwargs):
        Page.__init__(self, *args, **kwargs, bg="#36393f")

        self.read_json()
        acc_btn = tk.Button(self, command=self.setting_acc,
                            image=off_img, bg="#36393f")
        name_lb = tk.Label(self, text=self.name_list,
                           bg="#36393f", fg="#ffffff", font=("Times", 18))
        self.acc_btn = acc_btn

        name_lb.pack()
        acc_btn.pack()

    def read_json(self):
        with open(json_path, 'r', encoding="utf-8") as f:
            namelist = json.load(f)
        id_list = []
        name_list = []
        for num in namelist.values():
            id_list += [str(num['id'])]
            name_list += [str(num['name'])+","]
        self.name_list = name_list

    def setting_acc(self):
        global flag_acc
        flag_acc = not flag_acc
        if flag_acc:
            self.acc_btn['image'] = on_img
        else:
            self.acc_btn['image'] = off_img
        root.update()


class MainView(tk.Frame):

    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)

        p1 = Page1(self)
        p2 = Page2(self)
        p3 = Page3(self)

        buttonframe = tk.Frame(self, bg="#292b2f")
        container = tk.Frame(self)

        buttonframe.pack(side="top", fill="x", expand=False)
        container.pack(side="top", fill="both", expand=True)

        p1.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        p2.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        p3.place(in_=container, x=0, y=0, relwidth=1, relheight=1)

        b1 = tk.Button(buttonframe, text="主頁面", command=p1.lift,
                       bg="#36393f", fg="#ffffff")
        b2 = tk.Button(buttonframe, text="訓練頁面", command=p2.lift,
                       bg="#36393f", fg="#ffffff")
        b3 = tk.Button(buttonframe, text="查看資料", command=p3.lift,
                       bg="#36393f", fg="#ffffff")
        QUIT = tk.Button(buttonframe, text="QUIT", fg="red",
                         bg="#36393f", command=root.destroy)

        b1.pack(side="left")
        b2.pack(side="left")
        b3.pack(side="left")
        QUIT.pack(side="right")

        p1.show()


############################################################
if __name__ == "__main__":
    root = tk.Tk()
############################################################
    # 匯入檔案
# default setting之後可用.json
    flag_cam = False
    flag_acc = False
    flag_reset = False
    flag_limframe = False
    json_path = 'name.json'
    DataPath = 'Face'
############################################################
    on_img_op = Image.open("GUIimage/on.png")
    off_img_op = Image.open("GUIimage/off.png")
    on_img_op = on_img_op.resize((64, 64), Image.ANTIALIAS)
    off_img_op = off_img_op.resize((64, 64), Image.ANTIALIAS)
    on_img = ImageTk.PhotoImage(on_img_op)
    off_img = ImageTk.PhotoImage(off_img_op)
############################################################
    main = MainView(root)
    model = load_model('facenet_keras.h5')
    main.pack(side="top", fill="both", expand=True)
############################################################
    # 標題
    dict = {'title': '人臉辨識Face recognition', 'version': '3.1.1'}  # 標題，版本字典
    root.title(dict['title'] + '-' + dict['version'])
    # 大小
    root.geometry("1366x768")  # 寬x高
    root.minsize(width=960, height=540)
    # root.maxsize(width=1366, height=768)
    # ICON
    root.iconbitmap('GUIimage/bat.ico')
    # 透明度 0~1
    root.attributes('-alpha', 0.987654321)
############################################################
    root.mainloop()
