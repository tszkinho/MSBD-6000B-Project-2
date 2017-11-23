library(keras)
library(rio)
library(caret)
img_size<-c(200,200)
batch_size=32
epochs=300

options(keras.view_metrics=FALSE)

#load image data
train_path<-import("data/train.txt", header = FALSE)
train_img<-array(0,dim=c(nrow(train_path),img_size,3))
train_label<-to_categorical(train_path[,2])
val_path<-import("data/val.txt", header = FALSE)
val_img<-array(0,dim=c(nrow(val_path),img_size,3))
val_label<-to_categorical(val_path[,2])
test_path<-import("data/test.txt", header = FALSE)
test_img<-array(0,dim=c(nrow(test_path),img_size,3))

#preprocessing for training image
train_generator<-image_data_generator(rescale=1/255,
                                      shear_range = 0.1,
                                      zoom_range=0.1,
                                      horizontal_flip = TRUE,
                                      vertical_flip = TRUE,
                                      rotation_range = 90,
                                      width_shift_range = 0.2,
                                      height_shift_range = 0.2)

valid_generator<-image_data_generator(rescale=1/255)
test_generator<-image_data_generator(rescale=1/255)


#main network
base_model<-application_inception_v3(weights = "imagenet", include_top = FALSE)
# add dense layer
predictions<- base_model$output %>%
  layer_global_average_pooling_2d()%>%
  layer_dense(units=256)%>%
  layer_batch_normalization() %>%
  layer_activation(activation='relu') %>%
  layer_dense(units=5,activation='softmax')
model<-keras_model(inputs=base_model$input,outputs=predictions)
#freeze conv layer and train dense layer
freeze_layers(base_model)
model%>% compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics = "accuracy")


#load image to array
for( i in 1:nrow(train_path)){
  img<-image_load(paste("data",train_path[i,1],sep=""),target_size = img_size)
  train_img[i,,,]<-image_to_array(img)
  
}

for( i in 1:nrow(val_path)){
  img<-image_load(paste("data",val_path[i,1],sep=""),target_size = img_size)
  val_img[i,,,]<-image_to_array(img)
  
}


#fit the model with early stopping and reduce lr
history<-model %>%
  fit_generator(
    flow_images_from_data(train_img,train_label,train_generator,batch_size = batch_size),
    steps_per_epoch = as.integer(nrow(train_path)/batch_size),
    epochs=epochs,
    validation_data = flow_images_from_data(val_img,val_label,valid_generator,batch_size = batch_size),
    verbose=2,
    validation_steps = as.integer(nrow(val_path)/batch_size),
    callbacks=list(callback_reduce_lr_on_plateau(monitor="val_loss",factor=0.1),callback_early_stopping(monitor="val_loss",patience=20))
  )

#check result on validation image
evaluate(model,val_img/255,val_label)
p0<-predict(model,val_img/255)
confusionMatrix(apply(p0,1,which.max)-1,val_path[,2])
#save model for prediction
save_model_hdf5(model,"inceptionV3_200_layer128")



