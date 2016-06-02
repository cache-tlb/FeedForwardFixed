import google.protobuf as pb
import caffe_pb2
import random
from pbjson import *
import json

def convert_solver_state(file_name, dst_name):
    # Note: fixed point should not happen in solver state!
    f = open (file_name, 'rb')
    solver_state = caffe_pb2.SolverState()
    solver_state.ParseFromString(f.read())
    f.close()
    
    # do the change
    solver_state.iter = 0
    solver_state.learned_net = "caffe_voc_train_hft_iter_0"
    
    #for i in xrange(len(solver_state.history)):
        #num = solver_state.history[i].num
        #channel = solver_state.history[i].channels 
        #height = solver_state.history[i].height 
        #width = solver_state.history[i].width
        #total = (num * channel * height * width)
        #for j in xrange(total):
            #solver_state.history[i].data[j] = 0    
    
    # save
    s = solver_state.SerializeToString()
    f = open(dst_name, 'wb')
    f.write(s)
    f.close()

def convert_model_param(file_name, dst_name):
    f = open (file_name, 'rb')
    net_para = caffe_pb2.NetParameter()
    net_para.ParseFromString(f.read())
    f.close()
    
    # do the change
    net_para.name = "CaffeNet"
    net_para.layers[0].data_param.source = "voc_train_hft_leveldb"
    net_para.layers[0].data_param.mean_file = "../../data/ilsvrc12/imagenet_mean.binaryproto"
    
    # net_para.layers[0].top.append("label_count")
    # net_para.layers[0].top.append("label_vector")
    net_para.layers[0].top.append("image_id")
    net_para.layers[0].top.append("hypo_id")
    net_para.layers[0].top.append("hypo_count")
    
    # net_para.layers[23].bottom.append("label_count")
    # net_para.layers[23].bottom.append("label_vector")
    net_para.layers[23].bottom.append("image_id")
    net_para.layers[23].bottom.append("hypo_id")
    net_para.layers[23].bottom.append("hypo_count")
    net_para.layers[23].type = 32
    
    #n_labels = 20
    #net_para.layers[22].blobs[0].height = n_labels
    #num1 = net_para.layers[22].blobs[0].num
    #channels1 = net_para.layers[22].blobs[0].channels
    #height1 = net_para.layers[22].blobs[0].height
    #width1 = net_para.layers[22].blobs[0].width
    #n_weight = num1 * channels1 * height1 * width1
    #del net_para.layers[22].blobs[0].data[n_weight:]
    #for i in xrange(n_weight):
    #    net_para.layers[22].blobs[0].data[i] = random.gauss(0, 0.01)
    
    #net_para.layers[22].blobs[1].width = n_labels
    #num2 = net_para.layers[22].blobs[1].num
    #channels2 = net_para.layers[22].blobs[1].channels
    #height2 = net_para.layers[22].blobs[1].height
    #width2 = net_para.layers[22].blobs[1].width
    #n_weight = num2 * channels2 * height2 * width2
    #del net_para.layers[22].blobs[1].data[n_weight:]
    #for i in xrange(n_weight):
    #    net_para.layers[22].blobs[1].data[i] = random.gauss(0, 0.01)
    
    #save
    s = net_para.SerializeToString()
    f = open (dst_name, 'wb')
    f.write(s)
    f.close()

def show_pb(src, dst, model):
    f = open(src, 'rb')
    if model == 'model':
        x = caffe_pb2.NetParameter()
    elif model == 'state':
        x = caffe_pb2.SolverState()
    x.ParseFromString(f.read())
    f.close()    
    
    f = open (dst, 'w')
    print >> f, x
    f.close()
    
types = [ 'NONE', 
    'ACCURACY',
    'BNLL', # = 2;
    'CONCAT', # = 3;
    'CONVOLUTION', # = 4;
    'DATA', # = 5;
    'DROPOUT', # = 6;
    'EUCLIDEAN_LOSS', # = 7;
    'FLATTEN', # = 8;
    'HDF5_DATA' , #= 9;
    'HDF5_OUTPUT', # = 10;
    'IM2COL' , #= 11;
    'IMAGE_DATA', # = 12;
    'INFOGAIN_LOSS', # = 13;
    'INNER_PRODUCT', # = 14;
    'LRN' , #= 15;
    'MULTINOMIAL_LOGISTIC_LOSS', # = 16;
    'POOLING' , #= 17;
    'RELU' , #= 18;
    'SIGMOID' , #= 19;
    'SOFTMAX' , #= 20;
    'SOFTMAX_LOSS', # = 21;
    'SPLIT' , #= 22;
    'TANH', # = 23;
    'WINDOW_DATA', # = 24;
    'ELTWISE_PRODUCT', # = 25;
    'POWER' , #= 26;
    'SIGMOID_CROSS_ENTROPY_LOSS', # = 27;
    'HINGE_LOSS' , #= 28;
    'MEMORY_DATA' , #= 29;
    'SOFTMAX_LOSS_MULTILABEL', # = 30;
    'MULTILABEL_ACCURACY' , #= 31;
    'SOFTMAX_LOSS_HFT' , #= 32;
    'HFT_ACCURACY' , #= 33;
    'SOFTMAX_LOSS_FCN', # = 34
]

def convert_net_to_json(model_path, save_path):
    f = open (model_path, 'rb')
    net_para = caffe_pb2.NetParameter()
    net_para.ParseFromString(f.read())
    f.close()
    table = pb2dict(net_para)
    # for key in table:
    #     print key
    # print table['name']
    abs_min = 1e100
    v_min = 1e100
    v_max = -1e100
    cnt2 = 0
    ret = []
    print 'layer count:', len(table['layers'])
    for layer in table['layers']:
        layer_item = {}
        print 'layer name:', layer['name'], 'type:', types[layer['type']], '------------'
        # if 'blobs' not in layer:
            # continue
        # blob is a list contains a matrix and a bias
        if 'blobs' in layer:
            for item in layer['blobs']:
                print 'channel:', item['channels']
                print 'width:', item['width']
                print 'height:', item['height']
                print 'num:', item['num']
                print 'len of data', len(item['data'])
                cnt2 += item['num']*item['channels']*item['width']*item['height']
                # for v in item['data']:
                for k in range(len(item['data'])):
                    v = item['data'][k]
                    abs_min = min(abs_min, abs(v))
                    v_min = min(v_min, v)
                    v_max = max(v_max, v)
        if types[layer['type']] == 'CONVOLUTION':
            layer_item['type'] = types[layer['type']]
            layer_item['channels'] = layer['blobs'][0]['channels']
            layer_item['width'] = layer['blobs'][0]['width']
            layer_item['height'] = layer['blobs'][0]['height']
            layer_item['num'] = layer['blobs'][0]['num']
            layer_item['w'] = layer['blobs'][0]['data']
            layer_item['b'] = layer['blobs'][1]['data']
            layer_item['name'] = layer['name']
            pad = 0
            stride = 1
            if 'convolution_param' in layer:
                if 'pad' in layer['convolution_param']:
                    pad = layer['convolution_param']['pad']
                if 'stride' in layer['convolution_param']:
                    stride = layer['convolution_param']['stride']
            layer_item['pad'] = pad
            layer_item['stride'] = stride
            ret.append(layer_item)
        elif types[layer['type']] == 'INNER_PRODUCT':
            print layer['inner_product_param']
            layer_item['type'] = types[layer['type']]
            layer_item['channels'] = layer['blobs'][0]['channels']
            layer_item['num'] = layer['blobs'][0]['num']
            layer_item['width'] = layer['blobs'][0]['width']
            layer_item['height'] = layer['blobs'][0]['height']
            layer_item['w'] = layer['blobs'][0]['data']
            layer_item['b'] = layer['blobs'][1]['data']
            layer_item['name'] = layer['name']
            ret.append(layer_item)
        elif types[layer['type']] == 'POOLING':
            print layer['pooling_param']
            layer_item['type'] = types[layer['type']]
            layer_item['stride'] = layer['pooling_param']['stride']
            layer_item['size'] = layer['pooling_param']['kernel_size']
            policy = 'MAX'
            if 'pool' in layer['pooling_param']:
                policy = layer['pooling_param']['pool']
            layer_item['pool'] = policy
            layer_item['name'] = layer['name']
            ret.append(layer_item)
        elif types[layer['type']] == 'DATA':
            scale = 1
            if 'scale' in layer['data_param']:
                scale = layer['data_param']['scale']
            layer_item['scale'] = scale
            layer_item['name'] = layer['name']
            layer_item['type'] = types[layer['type']]
            ret.append(layer_item)
        else:
            layer_item['name'] = layer['name']
            layer_item['type'] = types[layer['type']]
            ret.append(layer_item)
    print 'abs_min:', abs_min
    print 'v_min:', v_min
    print 'v_max:', v_max
    print 'cnt:', cnt2
    if save_path != '':
        f = open(save_path, 'wb')
        json.dump(ret, f)
        f.close()
        # for key in layer:
            # print key
    # print type(table['layers'][0])
    # for key in table['layers'][0]:
    #     print key
    # print table['layers'][0]['window_data_param']
    # print table['layers'][0]['pooling_param']
    # print table['layers'][0]['layer']
    # print table['layers'][0]['top']
    # print table['layers'][0]['name']
    # print table['layers'][0]['type']
    # print table['layers'][0]['data_param']
    # f = open('temp.txt', 'w')
    # print >> f, table
    # f.close()
    
# convert_solver_state('caffe_voc_train_iter_0.solverstate', 'caffe_voc_train_hft_init.solverstate')
# show_pb('caffe_voc_train_hft_init.solverstate', 'caffe_voc_train_hft_init.solverstate.txt', 'state')
# show_pb('caffe_voc_train_hft_iter_20000.solverstate', 'caffe_voc_train_hft_iter_20000.solverstate.txt', 'state')

# convert_model_param('caffe_voc_train_iter_1500', 'caffe_voc_train_hft_init')
# show_pb('caffe_voc_train_hft_init', 'caffe_voc_train_hft_init.txt', 'model')
# show_pb('caffe_voc_train_iter_4000', 'caffe_voc_train_iter_4000.txt', 'model')
# show_pb('caffe_reference_imagenet_model', 'caffe_reference_imagenet_model.txt', 'model')
# show_pb('caffe_voc_train_hft_iter_20000', 'caffe_voc_train_hft_iter_20000.txt', 'model')
# show_pb('../../caffe-models/VGG_ILSVRC_16_layers.caffemodel', 'VGG_ILSVRC_16_layers.caffemodel.txt', 'model')
# convert_net_to_json('../../caffe-models/VGG_ILSVRC_16_layers.caffemodel', '')
convert_net_to_json('lenet_iter_5000', 'lenet_iter_5000.json')
# del net_para.layers[1].blobs[0].data[10:]

