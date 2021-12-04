from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


def stem(x0,num_classes):
    x = layers.Conv2D(32,3,strides=2,padding = "same",name="stem_conv")(x0)
    x = layers.Activation("elu",name="stem_activation")(x)

    return x
def block(x0,num_classes):
    count = "1"
    x = layers.DepthwiseConv2D(3,depth_multiplier=1,padding = "same",name="block"+count+"a_dwconv")(x0)
    print("tmp.shape=",x.shape.as_list())
    x = layers.Activation("elu",name="block"+count+"a_activation")(x)
    x = layers.Conv2D(16,1,padding = "same",name="block"+count+"a_project_conv")(x)


    tmp = layers.DepthwiseConv2D(3,depth_multiplier=1,padding = "same",name="block"+count+"b_dwconv")(x)
    tmp = layers.Activation("elu",name="block"+count+"b_activation")(tmp)
    tmp = layers.Conv2D(16,1,padding = "same",name="block"+count+"b_project_conv")(tmp)

    x = layers.add([tmp,x],name = "block"+count+"b_add")

    count = "2"
    x = layers.Conv2D(96,1,padding = "same",name="block"+count+"a_expand_conv")(x)
    x = layers.Activation("elu",name="block"+count+"a_expand_activation")(x)
    x = layers.DepthwiseConv2D(3,depth_multiplier=1,strides=2,padding = "same",name="block"+count+"a_dwconv")(x)
    x = layers.Activation("elu",name="block"+count+"a_activation")(x)
    x = layers.Conv2D(24,1,padding = "same",name="block"+count+"a_project_conv")(x)

    tmp = layers.Conv2D(144,1,padding = "same",name="block"+count+"b_expand_conv")(x)
    tmp = layers.Activation("elu",name="block"+count+"b_expand_activation")(tmp)
    tmp = layers.DepthwiseConv2D(3,depth_multiplier=1,padding = "same",name="block"+count+"b_dwconv")(tmp)
    tmp = layers.Activation("elu",name="block"+count+"b_activation")(tmp)
    tmp = layers.Conv2D(24,1,padding = "same",name="block"+count+"b_project_conv")(tmp)

    x = layers.add([tmp,x],name = "block"+count+"b_add")

    tmp = layers.Conv2D(144,1,padding = "same",name="block"+count+"c_expand_conv")(x)
    tmp = layers.Activation("elu",name="block"+count+"c_expand_activation")(tmp)
    tmp = layers.DepthwiseConv2D(3,depth_multiplier=1,padding = "same",name="block"+count+"c_dwconv")(tmp)
    tmp = layers.Activation("elu",name="block"+count+"c_activation")(tmp)
    tmp = layers.Conv2D(24,1,padding = "same",name="block"+count+"c_project_conv")(tmp)

    x = layers.add([tmp,x],name = "block"+count+"c_add")

    count="3"
    x = layers.Conv2D(144,1,padding = "same",name="block"+count+"a_expand_conv")(x)
    x = layers.Activation("elu",name="block"+count+"a_expand_activation")(x)
    x = layers.DepthwiseConv2D(5,depth_multiplier=1,strides=2,padding = "same",name="block"+count+"a_dwconv")(x)
    x = layers.Activation("elu",name="block"+count+"a_activation")(x)
    x = layers.Conv2D(48,1,padding = "same",name="block"+count+"a_project_conv")(x)

    tmp = layers.Conv2D(288,1,padding = "same",name="block"+count+"b_expand_conv")(x)
    tmp = layers.Activation("elu",name="block"+count+"b_expand_activation")(tmp)
    tmp = layers.DepthwiseConv2D(5,depth_multiplier=1,padding = "same",name="block"+count+"b_dwconv")(tmp)
    tmp = layers.Activation("elu",name="block"+count+"b_activation")(tmp)
    tmp = layers.Conv2D(48,1,padding = "same",name="block"+count+"b_project_conv")(tmp)

    x = layers.add([tmp,x],name = "block"+count+"b_add")

    tmp = layers.Conv2D(288,1,padding = "same",name="block"+count+"c_expand_conv")(x)
    tmp = layers.Activation("elu",name="block"+count+"c_expand_activation")(tmp)
    tmp = layers.DepthwiseConv2D(5,depth_multiplier=1,padding = "same",name="block"+count+"c_dwconv")(tmp)
    tmp = layers.Activation("elu",name="block"+count+"c_activation")(tmp)
    tmp = layers.Conv2D(48,1,padding = "same",name="block"+count+"c_project_conv")(tmp)

    x = layers.add([tmp,x],name = "block"+count+"c_add")

    count="4"

    x = layers.Conv2D(288,1,padding = "same",name="block"+count+"a_expand_conv")(x)
    x = layers.Activation("elu",name="block"+count+"a_expand_activation")(x)
    x = layers.DepthwiseConv2D(3,depth_multiplier=1,strides=2,padding = "same",name="block"+count+"a_dwconv")(x)
    x = layers.Activation("elu",name="block"+count+"a_activation")(x)
    x = layers.Conv2D(88,1,padding = "same",name="block"+count+"a_project_conv")(x)

    tmp = layers.Conv2D(528,1,padding = "same",name="block"+count+"b_expand_conv")(x)
    tmp = layers.Activation("elu",name="block"+count+"b_expand_activation")(tmp)
    tmp = layers.DepthwiseConv2D(3,depth_multiplier=1,padding = "same",name="block"+count+"b_dwconv")(tmp)
    tmp = layers.Activation("elu",name="block"+count+"b_activation")(tmp)
    tmp = layers.Conv2D(88,1,padding = "same",name="block"+count+"b_project_conv")(tmp)

    x = layers.add([tmp,x],name = "block"+count+"b_add")

    tmp = layers.Conv2D(528,1,padding = "same",name="block"+count+"c_expand_conv")(x)
    tmp = layers.Activation("elu",name="block"+count+"c_expand_activation")(tmp)
    tmp = layers.DepthwiseConv2D(3,depth_multiplier=1,padding = "same",name="block"+count+"c_dwconv")(tmp)
    tmp = layers.Activation("elu",name="block"+count+"c_activation")(tmp)
    tmp = layers.Conv2D(88,1,padding = "same",name="block"+count+"c_project_conv")(tmp)

    x = layers.add([tmp,x],name = "block"+count+"c_add")

    tmp = layers.Conv2D(528,1,padding = "same",name="block"+count+"d_expand_conv")(x)
    tmp = layers.Activation("elu",name="block"+count+"d_expand_activation")(tmp)
    tmp = layers.DepthwiseConv2D(3,depth_multiplier=1,padding = "same",name="block"+count+"d_dwconv")(tmp)
    tmp = layers.Activation("elu",name="block"+count+"d_activation")(tmp)
    tmp = layers.Conv2D(88,1,padding = "same",name="block"+count+"d_project_conv")(tmp)

    x = layers.add([tmp,x],name = "block"+count+"d_add")

    count = "5"

    x = layers.Conv2D(528,1,padding = "same",name="block"+count+"a_expand_conv")(x)
    x = layers.Activation("elu",name="block"+count+"a_expand_activation")(x)
    x = layers.DepthwiseConv2D(5,depth_multiplier=1,padding = "same",name="block"+count+"a_dwconv")(x)
    x = layers.Activation("elu",name="block"+count+"a_activation")(x)
    x = layers.Conv2D(120,1,padding = "same",name="block"+count+"a_project_conv")(x)

    tmp = layers.Conv2D(720,1,padding = "same",name="block"+count+"b_expand_conv")(x)
    tmp = layers.Activation("elu",name="block"+count+"b_expand_activation")(tmp)
    tmp = layers.DepthwiseConv2D(5,depth_multiplier=1,padding = "same",name="block"+count+"b_dwconv")(tmp)
    tmp = layers.Activation("elu",name="block"+count+"b_activation")(tmp)
    tmp = layers.Conv2D(120,1,padding = "same",name="block"+count+"b_project_conv")(tmp)

    x = layers.add([tmp,x],name = "block"+count+"b_add")

    tmp = layers.Conv2D(720,1,padding = "same",name="block"+count+"c_expand_conv")(x)
    tmp = layers.Activation("elu",name="block"+count+"c_expand_activation")(tmp)
    tmp = layers.DepthwiseConv2D(5,depth_multiplier=1,padding = "same",name="block"+count+"c_dwconv")(tmp)
    tmp = layers.Activation("elu",name="block"+count+"c_activation")(tmp)
    tmp = layers.Conv2D(120,1,padding = "same",name="block"+count+"c_project_conv")(tmp)

    x = layers.add([tmp,x],name = "block"+count+"c_add")

    tmp = layers.Conv2D(720,1,padding = "same",name="block"+count+"d_expand_conv")(x)
    tmp = layers.Activation("elu",name="block"+count+"d_expand_activation")(tmp)
    tmp = layers.DepthwiseConv2D(5,depth_multiplier=1,padding = "same",name="block"+count+"d_dwconv")(tmp)
    tmp = layers.Activation("elu",name="block"+count+"d_activation")(tmp)
    tmp = layers.Conv2D(120,1,padding = "same",name="block"+count+"d_project_conv")(tmp)

    x = layers.add([tmp,x],name = "block"+count+"d_add")

    count = "6"
    x = layers.Conv2D(720,1,padding = "same",name="block"+count+"a_expand_conv")(x)
    x = layers.Activation("elu",name="block"+count+"a_expand_activation")(x)
    x = layers.DepthwiseConv2D(5,depth_multiplier=1,strides = 2,padding = "same",name="block"+count+"a_dwconv")(x)
    x = layers.Activation("elu",name="block"+count+"a_activation")(x)
    x = layers.Conv2D(208,1,padding = "same",name="block"+count+"a_project_conv")(x)

    tmp = layers.Conv2D(1248,1,padding = "same",name="block"+count+"b_expand_conv")(x)
    tmp = layers.Activation("elu",name="block"+count+"b_expand_activation")(tmp)
    tmp = layers.DepthwiseConv2D(5,depth_multiplier=1,padding = "same",name="block"+count+"b_dwconv")(tmp)
    tmp = layers.Activation("elu",name="block"+count+"b_activation")(tmp)
    tmp = layers.Conv2D(208,1,padding = "same",name="block"+count+"b_project_conv")(tmp)

    x = layers.add([tmp,x],name = "block"+count+"b_add")

    tmp = layers.Conv2D(1248,1,padding = "same",name="block"+count+"c_expand_conv")(x)
    tmp = layers.Activation("elu",name="block"+count+"c_expand_activation")(tmp)
    tmp = layers.DepthwiseConv2D(5,depth_multiplier=1,padding = "same",name="block"+count+"c_dwconv")(tmp)
    tmp = layers.Activation("elu",name="block"+count+"c_activation")(tmp)
    tmp = layers.Conv2D(208,1,padding = "same",name="block"+count+"c_project_conv")(tmp)

    x = layers.add([tmp,x],name = "block"+count+"c_add")

    tmp = layers.Conv2D(1248,1,padding = "same",name="block"+count+"d_expand_conv")(x)
    tmp = layers.Activation("elu",name="block"+count+"d_expand_activation")(tmp)
    tmp = layers.DepthwiseConv2D(5,depth_multiplier=1,padding = "same",name="block"+count+"d_dwconv")(tmp)
    tmp = layers.Activation("elu",name="block"+count+"d_activation")(tmp)
    tmp = layers.Conv2D(208,1,padding = "same",name="block"+count+"d_project_conv")(tmp)

    x = layers.add([tmp,x],name = "block"+count+"d_add")

    tmp = layers.Conv2D(1248,1,padding = "same",name="block"+count+"e_expand_conv")(x)
    tmp = layers.Activation("elu",name="block"+count+"e_expand_activation")(tmp)
    tmp = layers.DepthwiseConv2D(5,depth_multiplier=1,padding = "same",name="block"+count+"e_dwconv")(tmp)
    tmp = layers.Activation("elu",name="block"+count+"e_activation")(tmp)
    tmp = layers.Conv2D(208,1,padding = "same",name="block"+count+"e_project_conv")(tmp)

    x = layers.add([tmp,x],name = "block"+count+"e_add")

    count = "7"

    x = layers.Conv2D(1248,1,padding = "same",name="block"+count+"a_expand_conv")(x)
    x = layers.Activation("elu",name="block"+count+"a_expand_activation")(x)
    x = layers.DepthwiseConv2D(3,depth_multiplier=1,padding = "same",name="block"+count+"a_dwconv")(x)
    x = layers.Activation("elu",name="block"+count+"a_activation")(x)
    x = layers.Conv2D(352,1,padding = "same",name="block"+count+"a_project_conv")(x)

    tmp = layers.Conv2D(2112,1,padding = "same",name="block"+count+"b_expand_conv")(x)
    tmp = layers.Activation("elu",name="block"+count+"b_expand_activation")(tmp)
    tmp = layers.DepthwiseConv2D(3,depth_multiplier=1,padding = "same",name="block"+count+"b_dwconv")(tmp)
    tmp = layers.Activation("elu",name="block"+count+"b_activation")(tmp)
    tmp = layers.Conv2D(352,1,padding = "same",name="block"+count+"b_project_conv")(tmp)

    x = layers.add([tmp,x],name = "block"+count+"b_add")

    return x

def top(x0,num_classes):
    x = layers.Conv2D(1408,1,padding = "same",name="top_conv")(x0)
    x = layers.Activation("elu",name="top_activation")(x)
    return x

def RNNnet(x,traffi_convection,desire,num_classes,rnn_state):
    desire1 = layers.Dense(use_bias=False, units = 8)(desire)
    traffi_convection1 = layers.Dense(use_bias=False, units = 2)(traffi_convection)
    x_concate = layers.Concatenate(axis=-1)([desire1, traffi_convection1, x])
    x_dense = layers.Dense(use_bias=False, units = 1024)(x_concate)
    x_1 = layers.Activation("relu")(x_dense)
    rnn_rz = layers.Dense(use_bias=False, units = 512)(rnn_state)
    rnn_rr = layers.Dense(use_bias=False, units = 512)(rnn_state)
    snpe_pleaser = layers.Dense(use_bias=False, units = 512)(rnn_state)
    rnn_rh = layers.Dense(use_bias=False, units = 512)(snpe_pleaser)

    rnn_z = layers.Dense(use_bias=False, units = 512)(x_1)
    rnn_h = layers.Dense(use_bias=False, units = 512)(x_1)
    rnn_r = layers.Dense(use_bias=False, units = 512)(x_1)

    add = layers.add([rnn_rz , rnn_z])
    activation_1 = layers.Activation("sigmoid")(add)
    add_1 = layers.add([rnn_rr , rnn_r])
    activation = layers.Activation("sigmoid")(add_1)

    multiply = rnn_rh*activation
    add_2 = layers.add([rnn_h , multiply])
    activation_2 = layers.Activation("tanh")(add_2)
    one_minus = layers.Dense(use_bias=False, units = 512)(activation_1)
    multiply_2 = one_minus*activation_2
    multiply_1 = snpe_pleaser*activation_1
    add_3 = layers.add([multiply_1 , multiply_2])
    return add_3

def posenet(x,traffi_convection,desire,num_classes,rnn_state):
    vf = layers.Flatten()(x)
    to_fk3 = RNNnet(vf,traffi_convection,desire,num_classes,rnn_state)
    fk1 = forks1(vf)
    fk2 = forks2(vf)
    fk3 = branch3(to_fk3)
    fk4 = branch4(to_fk3)
    outputs = layers.Concatenate(axis=-1)([fk1, fk2, fk3, fk4])
    return outputs

def forks1(x):
    x1 = layers.Dense(256, activation='relu', name="69meta")(x)
    meta = layers.Dense(4, activation='sigmoid', name="9meta")(x1)
    dp1 = layers.Dense(32, name="desire_final_dense")(x1)
    dp2 = layers.Reshape((4, 8), name="desire_reshape")(dp1)
    dp3 = layers.Softmax(axis=-1, name="desire_pred")(dp2)
    desire_pred = layers.Flatten(name="10desire_pred")(dp3)
    forks1_out = layers.Concatenate(axis=-1)([meta, desire_pred])
    return forks1_out

def forks2(x):
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(12, name="11pose")(x)
    return x

def branch3(x):
    xp = layers.Dense(256, activation='relu', name="1_path")(x)
    xp = layers.Dense(256, activation='relu', name="2_path")(xp)
    xp = layers.Dense(256, activation='relu', name="3_path")(xp)
    x1 = layers.Dense(128, activation='relu', name="final_path")(xp)
    xlx = layers.Dense(256, activation='relu', name="1_long_x")(x)
    xlx = layers.Dense(256, activation='relu', name="2_long_x")(xlx)
    xlx = layers.Dense(256, activation='relu', name="3_long_x")(xlx)
    x2 = layers.Dense(128, activation='relu', name="final_long_x")(xlx)
    xla = layers.Dense(256, activation='relu', name="1_long_a")(x)
    xla = layers.Dense(256, activation='relu', name="2_long_a")(xla)
    xla = layers.Dense(256, activation='relu', name="3_long_a")(xla)
    x3 = layers.Dense(128, activation='relu', name="final_long_a")(xla)
    xl = layers.Dense(256, activation='relu', name="1_lead")(x)
    xl = layers.Dense(256, activation='relu', name="2_lead")(xl)
    xl = layers.Dense(256, activation='relu', name="3_lead")(xl)
    x4 = layers.Dense(128, activation='relu', name="final_lead")(xl)
    out1 = layers.Dense(385, name="1path")(x1)
    out2 = layers.Dense(200, name="5long_x")(x2)
    out3 = layers.Dense(200, name="7long_a")(x3)
    out4 = layers.Dense(58, name="4lead")(x4)
    outputs = layers.Concatenate(axis=-1)([out1, out2, out3, out4])
    return outputs

def branch4(x):
    xll = layers.Dense(256, activation='relu', name="1_left_lane")(x)
    xll = layers.Dense(256, activation='relu', name="2_left_lane")(xll)
    xll = layers.Dense(256, activation='relu', name="3_left_lane")(xll)
    x5 = layers.Dense(128, activation='relu', name="final_left_lane")(xll)
    xlv = layers.Dense(256, activation='relu', name="1_long_v")(x)
    xlv = layers.Dense(256, activation='relu', name="2_long_v")(xlv)
    xlv = layers.Dense(256, activation='relu', name="3_long_v")(xlv)
    x6 = layers.Dense(128, activation='relu', name="final_long_v")(xlv)
    xds = layers.Dense(128, activation='relu', name="1_desire_state")(x)
    x7 = layers.Dense(8, name="final_desire_state")(xds)
    xrl = layers.Dense(256, activation='relu', name="1_right_lane")(x)
    xrl = layers.Dense(256, activation='relu', name="2_right_lane")(xrl)
    xrl = layers.Dense(256, activation='relu', name="3_right_lane")(xrl)
    x8 = layers.Dense(128, activation='relu', name="final_right_lane")(xrl)
    out5 = layers.Dense(386, name="2left_lane")(x5)
    out6 = layers.Dense(200, name="6long_v")(x6)
    out7 = layers.Softmax(axis=-1, name="8desire_state")(x7)
    out8 = layers.Dense(386, name="3right_lane")(x8)
    out9 = (x)
    outputs = layers.Concatenate(axis=-1)([out5, out6, out7, out8, out9])
    return outputs

def modell(x0,traffi_convection,desire,num_classes,rnn_state):
    x = stem(x0,num_classes)
    x = block(x,num_classes)
    x = top(x,num_classes)
    x = layers.Conv2D(32,1,padding = "same")(x)
    x = layers.Activation("elu")(x)
    x = posenet(x,traffi_convection,desire,num_classes,rnn_state)
    return x

def get_model(img_shape,traffi_convection_shape,desire_shape,rnn_state_shape, num_classes):
    inputs = keras.Input(shape=img_shape)

    x0 = layers.Permute((2,3,1))(inputs)
    traffi_convection = keras.Input(shape=traffi_convection_shape,name="traffi_convection")
    rnn_state = keras.Input(shape=rnn_state_shape,name="rnn_state")
    desire = keras.Input(shape=desire_shape,name="desire")

    path_output = modell(x0,traffi_convection,desire,num_classes,rnn_state)

    model = keras.Model(inputs = [inputs,traffi_convection,desire,rnn_state], outputs = path_output)
    return model


if __name__=="__main__":

    img_shape = (12, 128, 256)
    traffi_convection_shape = (2)
    desire_shape = (8)
    rnn_state_shape = (512)

    num_classes = 6
    model = get_model(img_shape,traffi_convection_shape,desire_shape,rnn_state_shape, num_classes)
    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_SUPAB.png')
    model.save('./saved_model/superAB.h5')
