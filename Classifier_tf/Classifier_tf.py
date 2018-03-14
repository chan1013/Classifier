import os
from tkinter import * 
from tkinter import ttk
from tkinter.filedialog import askdirectory
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from os import listdir
from os.path import isfile, isdir, join
import scipy
import PIL
from PIL import Image

from ReadImage import *
from model import *

class MainForm:
    def __init__(self, Form):
        self.learningrate =DoubleVar()
        self.TrainTime=IntVar()
        self.Dropout_rate=IntVar()
        self.Message=StringVar()
        self.path = StringVar()
        self.x_train=np.zeros((0,0,0,0))
        self.y_train=np.zeros((0,0,0,0))
        Form.title("Form")
        Form.geometry('800x600')
        
        self.path.set(r"C:/Users/jameschan/Source/Repos/Classifier/Classifier_tf/mnist/data")
        self.TrainTime.set(5)
        self.learningrate.set(0.01)
        self.Dropout_rate.set(0.5)

        #---------------------------訓練---------------------------
        #資料讀取
        self.lblPathWord=ttk.Label(Form,width=30, text="路徑:")
        self.lblPath=ttk.Label(Form,width=100, textvariable = self.path)#檔案路徑		
        self.btnSelectPath=ttk.Button(Form,width=2, text="...",command = self.btnSelectPath_Click)#更改路徑
        self.btnLoadData=ttk.Button(Form, width=7,text="載入資料", command=self.btnLoadData_Click)#載入資料
        self.lblDataNum=ttk.Label(Form, text="資料筆數:\n類型:")
        #訓練次數
        self.txtTrainTime = ttk.Entry(Form, width=7, textvariable=self.TrainTime)		
        self.lblTrainTime=ttk.Label(Form, text="訓練次數:")
        #學習率
        self.txtLearningRate = ttk.Entry(Form, width=7, textvariable=self.learningrate)		
        self.lblLearningRate=ttk.Label(Form, text="學習率:")
        #Dropout
        self.txtDropout_rate = ttk.Entry(Form, width=7, textvariable=self.Dropout_rate)		
        self.lblDropout_rate=ttk.Label(Form, text="Dropout:")
        #訓練過程資訊
        self.lblTrainMessage = ttk.Label(Form,width=100,justify = 'left',textvariable=self.Message)
        #開始訓練
        self.btnStarTrain=ttk.Button(Form,width=7,text="開始訓練", command=self.btnStarTrain_Click)


        ##版面編排
        TopX=20
        TopY=20
        SpaceX=60
        SpaceY=35
        #資料讀取
        self.lblPathWord.place(x=TopX,y=TopY-5)
        self.lblPath.place(x=TopX+30,y=TopY-5)
        self.btnSelectPath.place(x=TopX,y=TopY+SpaceY-10) 
        self.btnLoadData.place(x=TopX,y=TopY+SpaceY*2-15)  
        self.lblDataNum.place(x=TopX,y=TopY+SpaceY*3-20)
        #訓練次數
        self.lblTrainTime.place(x=TopX,y=TopY+SpaceY*4)
        self.txtTrainTime.place(x=TopX+SpaceX,y=TopY+SpaceY*4)
        #學習率
        self.lblLearningRate.place(x=TopX,y=TopY+SpaceY*5)
        self.txtLearningRate.place(x=TopX+SpaceX,y=TopY+SpaceY*5)
        #Dropout
        self.lblDropout_rate.place(x=TopX,y=TopY+SpaceY*6)
        self.txtDropout_rate.place(x=TopX+SpaceX,y=TopY+SpaceY*6)
        #訓練過程資訊
        self.lblTrainMessage.place(x=TopX,y=TopY+SpaceY*7)
        self.btnStarTrain.place(x=300,y=TopY+SpaceY*6)
    

    
    
#----------介面按鈕----------	
    def btnSelectPath_Click(self):
        path_ = askdirectory()
        self.path.set(path_)

    def btnLoadData_Click(self):

        #self.LoadMNIST(self.path.get())
        self.LoadImage(self.path.get())

    def btnStarTrain_Click(self):

        self.Train(img_size=self.x_train.shape[1],num_classes=self.y_train.shape[1],num_steps=self.TrainTime.get(),display_step=1)

#----------function----------
    def LoadImage(self,path):
        
        self.FileName,self.ImageData=ImageFileToArray(path)
        ImageSize=self.ImageData.shape[1]

        if(self.ImageData.ndim==3):##grayscale
            ImageType="灰階"
            ImageDepth=1
        if(self.ImageData.ndim==4):##grayscale
            ImageType="彩色"
            ImageDepth=3


        x_train=np.reshape(self.ImageData,[-1,ImageSize,ImageSize,ImageDepth])
        self.x_train=x_train/255
        self.y_train=make_label(self.FileName)        
        self.lblDataNum.configure(text="資料筆數:{}\n類型:{}".format(self.ImageData.shape[0],ImageType))


    def LoadMNIST(self,path):
        
        filename="mnist.npz"
        filepath=join(path,filename)
        f = np.load(filepath)
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        f.close()
        x_train=np.reshape(x_train,[-1,28,28,1])
        self.x_train=x_train/ 255
        self.y_train = Onehot(y_train)   

        self.lblDataNum.configure(text="資料筆數:{}".format(x_train.shape[0]) )

    def Train(self,img_size,num_classes,num_steps,display_step = 100,dropout=0.5,batch_size=100,learning_rate=0.01):
        x_train=self.x_train
        y_train=self.y_train
        X = tf.placeholder(tf.float32, [None, img_size,img_size,1],name='input')
        Y = tf.placeholder(tf.float32, [None, num_classes])
        keep_prob = tf.placeholder(tf.float32,name='keep_prob') # dropout (keep probability)
        weights ={
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 16])),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([5, 5, 16, 32])),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([7*7*32, 1024])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, num_classes]))
        }

        biases = {
            'bc1': tf.Variable(tf.random_normal([16])),
            'bc2': tf.Variable(tf.random_normal([32])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([num_classes]))
        }

        # Construct model
        logits = conv_net(X, weights, biases, keep_prob)
        prediction = tf.nn.softmax(logits,name='output')

        # Define loss and optimizer
        loss_op = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)


        # Evaluate model
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32),name="accuracy")

        # Initialize the variables
        init = tf.global_variables_initializer()

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        # 訓練
        message=""
        for step in range(1, num_steps+1):
            offset = (step * batch_size) % (x_train.shape[0] - batch_size)
            batch_data = x_train[offset:(offset + batch_size), :]
            batch_labels=y_train[offset:(offset + batch_size), :]

            sess.run(train_op, feed_dict={X: batch_data, Y: batch_labels,keep_prob: dropout})
            if step % display_step == 0 or step == 1:
                # 訓練時印出結果
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_data,Y: batch_labels,keep_prob: 1.0})
                #print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))
                message="Step:{} Minibatch Loss={:0.2f} Training Accuracy={:0.2f}\r".format(step,loss,acc)            
                self.Message.set(message)
                #lblTrainMessage.configure(text=message) 
                self.lblTrainMessage.update()

        #print("訓練完成")
        message+="訓練完成"
        #lblTrainMessage.configure(text=message)
        self.Message.set(message)
        output_graph_def = convert_variables_to_constants(sess, sess.graph_def, output_node_names=["output"])
        with tf.gfile.FastGFile("./mnist/MnistCnn.pb", mode='wb') as f:
            f.write(output_graph_def.SerializeToString())
Form = Tk()
GUI = MainForm(Form)
Form.mainloop()