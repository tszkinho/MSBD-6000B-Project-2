library(keras)
library(rio)
library(caret)
img_size<-c(200,200)
batch_size=32
epochs=300

options(viewer=NULL)
#load image data
train_path<-import("data/train.txt", header = FALSE)
train_img<-array(0,dim=c(nrow(train_path),img_size,3))
train_label<-to_categorical(train_path[,2])
val_path<-import("data/val.txt", header = FALSE)
val_img<-array(0,dim=c(nrow(val_path),img_size,3))
val_label<-to_categorical(val_path[,2])
test_path<-import("data/test.txt", header = FALSE)
test_img<-array(0,dim=c(nrow(test_path),img_size,3))

#load image to array
for( i in 1:nrow(val_path)){
  img<-image_load(paste("data",val_path[i,1],sep=""),target_size = img_size)
  val_img[i,,,]<-image_to_array(img)
  
}

for( i in 1:nrow(test_path)){
  img<-image_load(paste("data",test_path[i,1],sep=""),target_size = img_size)
  test_img[i,,,]<-image_to_array(img)
  
}

#load model
model<-load_model_hdf5("inceptionV3_200_layer256")

#check result on validation image
evaluate(model,val_img/255,val_label)
p0<-predict(model,val_img/255)
confusionMatrix(apply(p0,1,which.max)-1,val_path[,2])

#predict on test image
p1<-predict(model,test_img/255)
test_label<-data.frame(apply(p1,1,which.max)-1)
export(test_label,"project2_20392904.txt", col.names = FALSE)