# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 22:08:27 2020

@author: voyno
"""


from keras import Model
from keras.layers import Conv2D, AvgPool2D, Flatten, Dense, BatchNormalization, Input, Add, Activation
from keras.preprocessing import image


class ResNet:
    
    
    def __init__(self, num_layers=20):
        self.stack_size = (num_layers - 2) // 6
        self.model = self.build(self.stack_size)
        self.model.compile(optimizer='adam', loss='mse', metrics=['acc'])
        
    
    def train(self, train, test, batch_size=256, epochs=200, verbose=2, augment=False):
        
        """
        Parameters
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        train       tuple : x_train and y_train data
        tests       tuple : x_test and y_test data
        batch_size  int   : number of inputs per gradient update
        epochs      int   : number of iterations of training
        verbose     bool  : if True training output else no ouput
        argument    bool  : if true use data augmentation
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        """
        
        if augment:
            datagen = image.ImageDataGenerator(
                    rotation_range=30,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.1,
                    zoom_range=0.1,
                    horizontal_flip=True)
            
            return self.model.fit_generator(datagen.flow(train[0], train[1], batch_size=batch_size), 
                                            steps_per_epoch=len(train[0])//batch_size,
                                            epochs=epochs,
                                            verbose=verbose,
                                            validation_data=(test[0], test[1]))
        
        else:
           return self.model.fit(train[0],
                                 train[1],
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 verbose=verbose,
                                 validation_data=(test[0], test[1]))
           
       
    def build(self, stack_size, summary=False):
        
        """
        Parameters
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        stack_size  int  : number of layers per filter_size stack
        summary     bool : display model summary if true
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        """
        
        input_shape=(32, 32, 3)
        num_filter = 16
        num_stacks = 3
        
        x_in = Input(shape=input_shape)
        x = Conv2D(num_filter, kernel_size=3, padding='same', activation='relu')(x_in)
        
        for i in range(num_stacks):
            for j in range(stack_size):
                if i != 0 and j == 0:
                    x = self.conv_block(x, num_filter, projection=True)
                else:
                    x = self.conv_block(x, num_filter)
            
            num_filter *= 2
            
        x = AvgPool2D(8)(x)
        x = Flatten()(x)
        x_out = Dense(10)(x)
        model = Model(inputs=x_in, outputs=x_out)
        
        if summary:
            print(model.summary())
        
        return model

    
    def conv_block(self, x, num_filter, projection=False):
        
        """
        Parameters
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        x           tensor: output from previous conv layer
        num_filter  int   : number of filters for conv layer
        projection  bool  : logic for 1x1 conv on residual path
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        """
    
        x_resid = x
        
        if projection:
            x_resid = self.conv_layer(x_resid, num_filter, kernel_size=1, strides=2)
            
            x = self.conv_layer(x, num_filter, strides=2)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
        else:
            x = self.conv_layer(x, num_filter)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
        
        x = self.conv_layer(x, num_filter)
        x = BatchNormalization()(x)
        
        x = Add()([x, x_resid])
        x = Activation('relu')(x)
        
        return x


    @staticmethod
    def conv_layer(inputs, filters, kernel_size=3, strides=1):
        
        """
        Parameters
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        inputs      tensor: output from previous conv layer
        filters     int   : number of filters for conv layer
        kernel_size int   : size of sliding conv filter (n x n)
        strides     int   : size of movement per filter slide
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        """
    
        x = Conv2D(filters=filters, 
                   kernel_size=kernel_size, 
                   strides=strides, 
                   padding='same')(inputs)

        return x
    
