import caffe
from caffe import layers as L, params as P

def FractalNet(lmdb, batch_size):
    # My version of the 3D Fractal Net
    n = caffe.NetSpec()

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, \
    source=lmdb, transform_param=dict(scale=1./255), ntop=2)

    # Layer 1
    n.conv1 = L.Convolution(n.data, kernel_size=3, num_output=64, \
    weight_filler=dict(type='gaussian', std=0.01), \
    bias_filler=dict(type='constant',value=0))
    n.batch_norm1 = L.BatchNorm(n.conv1, in_place=True, param=[dict(lr_mult=0, \
    decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    n.relu1 = L.ReLU(n.batch_norm1, in_place=True)

    # Layer 2
    n.pool1 = L.Pooling(n.relu1, kernel_size=3, stride=2, pool=P.Pooling.MAX)

    # Layer 3
    n.conv2 = L.Convolution(n.pool1, kernel_size=3, num_output=128, \
    weight_filler=dict(type='gaussian', std=0.01), \
    bias_filler=dict(type='constant',value=0))
    n.batch_norm2 = L.BatchNorm(n.conv2, in_place=True, param=[dict(lr_mult=0, \
    decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    n.relu2 = L.ReLU(n.batch_norm2, in_place=True)

    # Layer 4
    n.pool2 = L.Pooling(n.relu2, kernel_size=3, stride=2, pool=P.Pooling.MAX)

    # Layer 5
    n.conv3 = L.Convolution(n.pool2, kernel_size=3, num_output=256, \
    weight_filler=dict(type='gaussian', std=0.01), \
    bias_filler=dict(type='constant',value=0))
    n.batch_norm3 = L.BatchNorm(n.conv3, in_place=True, param=[dict(lr_mult=0, \
    decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    n.relu3 = L.ReLU(n.batch_norm3, in_place=True)

    # Layer 6
    n.pool3 = L.Pooling(n.relu3, kernel_size=3, stride=2, pool=P.Pooling.MAX)

    # Layer 7
    n.conv4 = L.Convolution(n.pool3, kernel_size=3, num_output=512, \
    weight_filler=dict(type='gaussian', std=0.01), \
    bias_filler=dict(type='constant',value=0))
    n.batch_norm4 = L.BatchNorm(n.conv4, in_place=True, param=[dict(lr_mult=0, \
    decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    n.relu4 = L.ReLU(n.batch_norm4, in_place=True)

    # Layer 8
    n.deconv1 = L.Deconvolution(n.relu4, convolution_param=dict(num_output=256, \
    kernel_size=3, stride=2, bias_term=False),param=[dict(lr_mult=0)])

    # Layer 9
    n.eltsum1 = L.Eltwise(n.deconv1,n.relu3, coeff=1)

    # Layer 10
    n.conv5 = L.Convolution(n.eltsum1, kernel_size=3, num_output=256, \
    weight_filler=dict(type='gaussian', std=0.01), \
    bias_filler=dict(type='constant',value=0))
    n.batch_norm5 = L.BatchNorm(n.conv5, in_place=True, param=[dict(lr_mult=0, \
    decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    n.relu5 = L.ReLU(n.batch_norm5, in_place=True)

    # Layer 11
    n.deconv2 = L.Deconvolution(n.relu5, convolution_param=dict(num_output=128, \
    kernel_size=3, stride=2, bias_term=False),param=[dict(lr_mult=0)])

    # Layer 12
    n.eltsum2 = L.Eltwise(n.deconv2,n.relu2, coeff=1)

    # Layer 13
    n.conv6 = L.Convolution(n.eltsum2, kernel_size=3, num_output=128, \
    weight_filler=dict(type='gaussian', std=0.01), \
    bias_filler=dict(type='constant',value=0))
    n.batch_norm6 = L.BatchNorm(n.conv6, in_place=True, param=[dict(lr_mult=0, \
    decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    n.relu6 = L.ReLU(n.batch_norm6, in_place=True)

    # Layer 14
    n.deconv3 = L.Deconvolution(n.relu6, convolution_param=dict(num_output=64, \
    kernel_size=3, stride=2, bias_term=False),param=[dict(lr_mult=0)])

    # Layer 15
    n.eltsum3 = L.Eltwise(n.deconv3,n.relu1, coeff=1)

    # Layer 16
    n.conv7 = L.Convolution(n.eltsum3, kernel_size=3, num_output=64, \
    weight_filler=dict(type='gaussian', std=0.01), \
    bias_filler=dict(type='constant',value=0))
    n.batch_norm7 = L.BatchNorm(n.conv7, in_place=True, param=[dict(lr_mult=0, \
    decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    n.relu7 = L.ReLU(n.batch_norm7, in_place=True)

    # Layer 17
    n.softmax1 = L.Softmax(n.relu7)

    # Layer 18
    n.conv8 = L.Convolution(n.softmax1, kernel_size=1, num_output=64, \
    weight_filler=dict(type='gaussian', std=0.01), \
    bias_filler=dict(type='constant',value=0))

    return n.to_proto()

with open('fractal_train_lmdb.prototxt', 'w') as f:
    f.write(str(FractalNet('fractal_train_lmdb', 100)))

with open('fractal_test_lmdb.prototxt', 'w') as f:
    f.write(str(FractalNet('fractal_test_lmdb', 100)))
