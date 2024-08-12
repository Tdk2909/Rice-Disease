import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import to_categorical #######'''

# Tạo một đối tượng để chỉ địa chỉ ảnh
diaChiAnhTrain='C:/Users//PC/.spyder-py3/datasets/train-data'
# khai báo lớp imagedatagenerator, đây là lớp tăng cường dữ liệu
from keras.preprocessing.image import ImageDataGenerator
#Tạo một đối tượng chứa dữ liệu đầu vào tăng cường 
train_tangcuong = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)

#tương tự vậy tạo 1 đối tượng kiểm thử (test) 
test_tangcuong = ImageDataGenerator()

# Tạo một đối tượng dùng để train cho mô hình, đối tượng này đã được tăng cường dữ liệu
# và lấy dữ liệu nạp lên từ đối tượng địa chỉ hình ảnh
dulieu_train = train_tangcuong.flow_from_directory(
        diaChiAnhTrain,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')


#Tương tự tạo 1 đối tượng test
dulieu_test = test_tangcuong.flow_from_directory(
        diaChiAnhTrain,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

# Gán nhẫn cho dữ liệu test
dulieu_test.class_indices

'''############ tạo lớp chứa các nhãn và giá trị số của từng loại bệnh ############'''
# Tạo một đối tượng để lưu các nhãn đã gắn cho dữ liệu train, 
# Hàm class_indices sẽ giúp trả về giá trị nhãn của đối tượng dữ liệu train
lop_train=dulieu_train.class_indices

# tạo một đối tượng ResultMap để lưu các nhãn số và tên bệnh ở lúa tương ứng.
ResultMap={}
for gtNhan,tenNhan in zip(lop_train.values(),lop_train.keys()):
    ResultMap[gtNhan]=tenNhan

#lưu đối tượng ResultMap vào tệp ResultsMap.pkl để sử dụng sau này.
import pickle
with open("ResultsMap.pkl", 'wb') as fileWriteStream:
    pickle.dump(ResultMap, fileWriteStream)

# In đối tượng ResultMap
print("giá trị nhãn và tên bệnh tương ứng: ",ResultMap)

# các loại bệnh có thể dự đoán phải bằng số lương bệnh mình đã train
du_doan_benh=len(ResultMap)
print('\n Số lượng bệnh : ', du_doan_benh)


'''######################## XÂY DỰNG MÔ HÌNH CNN ########################'''
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

# tạo một đối tượng mohinh có những thuộc tính của hàm Sequential, đối tượng mohinh sẽ dùng để chứa các đặc tính của mô hình CNN
# lớp Sequential là một lớp mô hình tuần tự (sequential model) từ đó có thể thêm các lớp(đặc tính) vào trong Sequential() 
mohinh= Sequential()
#Thêm một lớp Convolution 2D vào mô hình. lớp này dùng để xử lí hình ảnh, chỉ có 2 chiều
mohinh.add(Convolution2D(32, kernel_size=(7, 7), strides=(1, 1), input_shape=(64,64,3),
                         activation='relu'))
#Thêm một lớp MaxPool2D vào mô hình, lớp này dùng để chọn lọc ra các giá trị đặc trưng sau khi đi qua lớp Convolution2D
mohinh.add(MaxPool2D(pool_size=(2,2)))
#Tiếp tục nâng cấp lớp Conv2D với số lượng kernel = 64 (>32) để mô hình chính xác hơn 
mohinh.add(Convolution2D(64, kernel_size=(7, 7), strides=(1, 1), activation='relu'))
mohinh.add(MaxPool2D(pool_size=(2,2)))
#Tiếp tục nâng cấp lớp Conv2D với số lượng kernel = 128 (>64) để mô hình chính xác hơn 
mohinh.add(Convolution2D(128, kernel_size=(7, 7), strides=(1, 1), activation='relu'))
mohinh.add(MaxPool2D(pool_size=(2,2)))
#Flatten mô hình
mohinh.add(Flatten())
#Tạo lớp fully connected 
mohinh.add(Dense(128, activation='relu'))
mohinh.add(Dense(du_doan_benh, activation='softmax'))
#sử dụng câu lệnh compile() để biên dịch mô hình CNN
# và dùng các hàm khác để huấn luyện và đánh giá mô hình
mohinh.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=["accuracy"])
#Tính toán thời gian huấn luyện mô hình
import time
StartTime=time.time()
# Huấn luyện mô hình
mohinh.fit_generator(
                    dulieu_train,
                    steps_per_epoch=5,
                    epochs=12,
                    validation_data=dulieu_test,
                    validation_steps=10)

EndTime=time.time()
print("###### Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes ######')


# Chương trình đưa ra dự đoán về bệnh mà cây lúa mắc phải sau khi huấn luyện
import numpy as np
import keras.utils as image
import tensorflow as tf

#Tạo 1 đối tượng dẫn đến địa chỉ hình ảnh dùng để test
ImagePath='C:/Users/PC/.spyder-py3/datasets/test-data/chuaqua_daotao/test1.jpg'

#tạo đối tượng chứa hình ảnh được tải lên dùng để test
test_hinhanh=tf.keras.utils.load_img(ImagePath,target_size=(64, 64))
#Đối tượng hình ảnh dùng để test được chuyển về mảng array
test_hinhanh=tf.keras.utils.img_to_array(test_hinhanh)
#Trước khi sử dụng mảng NumPy test_hinhanh làm đầu vào cho mô hình CNN,chúng ta cần thêm
#một chiều mới vào mảng này để phù hợp với định dạng đầu vào của mô hình CNN.
test_hinhanh=np.expand_dims(test_hinhanh,axis=0)
#dự đoán nhãn của hình ảnh test_hinhanh.
#Hàm predict() của mô hình CNN trả về một mảng NumPy có chứa các xác suất dự đoán cho từng nhãn.
result=mohinh.predict(test_hinhanh,verbose=0)

#In ra dự đoán
print('________________________________________________--')
print('Dự đoán là is: ',ResultMap[np.argmax(result)])