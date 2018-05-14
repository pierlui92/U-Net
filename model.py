from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
from six.moves import xrange
from collections import namedtuple
from scipy import misc
import datetime

from module import *
from utils import color

class U_Net(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size

        self.crop_size_w = args.crop_size_w
        self.crop_size_h = args.crop_size_h
        # self.load_size_w = args.load_size_w
        # self.load_size_h = args.load_size_h

        self.input_c_dim = args.input_nc

        self.with_flip=args.flip

        self.num_sample = args.num_sample
        self.num_sample_test = args.num_sample_test
        self.num_epochs = args.epoch
        self.num_classes = args.num_classes

        self.input_list_train = args.input_list_train
        self.input_list_val_test = args.input_list_val_test

        self.criterionSem = sem_criterion
        self.best = 0

        OPTIONS = namedtuple('OPTIONS', 'gf_dim num_classes')
        self.options = OPTIONS._make((args.ngf,self.num_classes))

        if args.phase=='train':
            self._build_model()
            self.saver = tf.train.Saver(max_to_keep=2, keep_checkpoint_every_n_hours=2)
            self.saver_best = tf.train.Saver(max_to_keep=2)

    def _build_model(self):
        immy_a,_ ,_,immy_a_sem= self.build_input_image_op(self.input_list_train,False)
        self.input_images, self.input_sem_gt = tf.train.shuffle_batch([immy_a,immy_a_sem],self.batch_size,100,30,8)
                 
        self.input_sem_pred = u_net_model(self.input_images,self.options, False, name = 'u_net')
        self.sem_loss =  self.criterionSem(self.input_sem_pred,self.input_sem_gt,self.num_classes)
        self.sem_loss_sum = tf.summary.scalar("sem_loss",self.sem_loss,collections=["TRAINING_SCALAR"])

        immy_val,path_val,_,immy_val_sem = self.build_input_image_op(self.input_list_val_test,is_val=True)

        self.val_images,self.val_path, self.val_sem_gt = [tf.expand_dims(immy_val,axis=0),tf.expand_dims(path_val,axis=0),tf.expand_dims(immy_val_sem,axis=0)]
        self.val_sem_pred = u_net_model(self.val_images,self.options, True, name = 'u_net',is_test=True)
        
        ignore_label = 255
        weightsValue = tf.to_float(tf.not_equal(self.val_sem_gt,ignore_label))
        
        self.val_sem_pred = tf.expand_dims(tf.argmax(self.val_sem_pred,axis=-1),axis=-1)
        self.total = tf.reduce_sum(tf.multiply(tf.to_float(tf.equal(tf.cast(self.val_sem_gt,tf.float32),tf.cast(self.val_sem_pred,tf.float32))),weightsValue))
        self.count = tf.reduce_sum(weightsValue)
        
        self.accuracy_placeholder = tf.placeholder(tf.float32)
        self.val_sem_accuracy_sum = tf.summary.scalar("accuracy",self.accuracy_placeholder, collections=["VALIDATION_SCALAR"])            

    def build_input_image_op(self,input_list_txt,is_test=False,is_val=False, num_epochs=None):
        def _parse_function(image_tensor):
            image = tf.read_file(image_tensor[0])
            image_sem = tf.read_file(image_tensor[1])
            image = tf.image.decode_image(image, channels = 3)
            image_sem = tf.image.decode_image( image_sem , channels = 1)
            image.set_shape([None,None,3])
            image_sem.set_shape([None,None,1])
            return image , image_tensor[0], image_sem
        
        samples=[]
        samples_sem = []
        
        with open(input_list_txt, "r") as input_list:
            for line in input_list:
                sample,sample_sem = line.strip().split(";")
                samples.append(sample.strip())
                samples_sem.append(sample_sem.strip())

        inputs = np.stack((samples, samples_sem), axis = -1)
        
        image_tensor = tf.constant(inputs, tf.string)
        dataset = tf.data.Dataset.from_tensor_slices(image_tensor)
        dataset = dataset.map(_parse_function)
        dataset = dataset.repeat()
        iterator = dataset.make_one_shot_iterator()
        
        image , image_path, image_sem = iterator.get_next()
        im_shape= tf.shape(image)

        #change range of value o [-1,1]
        image = tf.image.convert_image_dtype(image,tf.float32)
        image = (image*2)-1
        if not is_test and not is_val:
            #resize to load_size
            # image = tf.image.resize_images(image,[self.load_size_h,self.load_size_w])
            # image_sem = tf.image.resize_images(image_sem, [self.load_size_h,self.load_size_w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
              
            crop_offset_w = tf.cond(tf.equal(tf.shape(image)[1]- self.crop_size_w,0), lambda : 0, lambda : tf.random_uniform((), minval=0, maxval= tf.shape(image)[1]- self.crop_size_w, dtype=tf.int32))
            crop_offset_h = tf.cond(tf.equal(tf.shape(image)[0]- self.crop_size_h,0), lambda : 0, lambda : tf.random_uniform((), minval=0, maxval= tf.shape(image)[0]- self.crop_size_h, dtype=tf.int32))
            image = tf.image.crop_to_bounding_box(image, crop_offset_h, crop_offset_w, self.crop_size_h, self.crop_size_w)          
            image_sem = tf.image.crop_to_bounding_box(image_sem, crop_offset_h, crop_offset_w, self.crop_size_h, self.crop_size_w)          
            image.set_shape([None,None,3])
            image_sem.set_shape([None,None,1])

            #random flip left right
            # if self.with_flip:
            #     image = tf.image.random_flip_left_right(image)
        elif is_test:
            tmp = (256  - im_shape[0] % 256) % 256
            pad_up = tmp//2
            pad_down =  tmp -  tmp//2
            tmp = (256  - im_shape[1] % 256) % 256
            pad_left = tmp//2
            pad_right =  tmp -  tmp//2
            image = tf.pad(image,[[pad_up,pad_down],[pad_left,pad_right],[0,0]],"REFLECT")
            image_sem = tf.pad(image_sem,[[pad_up,pad_down],[pad_left,pad_right],[0,0]],"REFLECT")
            image.set_shape([None,None,3])
            image_sem.set_shape([None,None,1])
        #     image = tf.image.resize_images(image,[self.load_size_h,self.load_size_w])
        #     image_sem = tf.image.resize_images(image_sem, [self.load_size_h,self.load_size_w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return image,image_path,im_shape, image_sem
    
    def accuracy_validation(self,args):
        total_loc = 0
        count_loc = 0
        for idx in range(self.num_sample_test):
            print("Evaluating accuracy on validation set", idx + 1 ,"/", self.num_sample_test, end= '\r' if idx != self.num_sample_test - 1 else '\n')
            total_value, count_value = self.sess.run([self.total,self.count])
            total_loc += total_value
            count_loc += count_value
        print(total_loc,count_loc)
        mean_acc = total_loc / count_loc
        return mean_acc

    def train(self, args):
        """Train cyclegan"""
        self.u_net_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
            .minimize(self.sem_loss)
        
        #summaries for training
        in_pred = tf.expand_dims(tf.argmax(self.input_sem_pred , axis=-1), axis =-1)
        val_pred = self.val_sem_pred
        
        tf.summary.image('train',self.input_images,max_outputs=1,collections=["TRAINING_IMAGES"])
        tf.summary.image('train sem gt',color(self.input_sem_gt),max_outputs=1,collections=["TRAINING_IMAGES"])
        tf.summary.image('train sem pred', color(in_pred),max_outputs=1,collections=["TRAINING_IMAGES"])
    
        tf.summary.image('val',self.val_images,max_outputs=1,collections=["VALIDATION_IMAGES"])
        tf.summary.image('val sem gt',color(self.val_sem_gt),max_outputs=1,collections=["VALIDATION_IMAGES"])
        tf.summary.image('val sem pred', color(val_pred),max_outputs=1,collections=["VALIDATION_IMAGES"])

        summary_scalar_train_op = tf.summary.merge(tf.get_collection("TRAINING_SCALAR"))
        summary_scalar_val_op = tf.summary.merge(tf.get_collection("VALIDATION_SCALAR"))
        summary_images_train_op = tf.summary.merge(tf.get_collection("TRAINING_IMAGES"))
        summary_images_val_op = tf.summary.merge(tf.get_collection("VALIDATION_IMAGES"))
        
        init_op = [tf.global_variables_initializer(),tf.local_variables_initializer()]
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter(args.checkpoint_dir, self.sess.graph)
        
        self.counter = 0
        
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners()
        print('Thread running')
        
        if self.load(os.path.join(args.checkpoint_dir,"best")):
            print("Evaluating old best accuracy on validation set")
            self.best=self.accuracy_validation(args)
            print("Old best accuracy: ", self.best)
        else:
            print("Old best checkpoint not found")
        
        print("Loading last checkpoint")
        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        
        total_time= 0
        start_iter = self.counter

        while self.counter <= self.num_epochs*self.num_sample:
            # Update network
            start_time = time.time()
            loss , _ = self.sess.run([self.sem_loss, self.u_net_optim])
            
            
            if (self.counter // self.num_sample) < ((self.counter + self.batch_size)//self.num_sample) and self.counter !=0:
                mean_acc = self.accuracy_validation(args)
                if mean_acc >= self.best:      
                    print("Accuracy step ", self.counter, ": " , mean_acc, " old best: " , self.best)             
                    self.best=mean_acc
                    self.save(self.saver_best,os.path.join(args.checkpoint_dir,"best"),self.counter)

                #### summary writing ####
                summary_string = self.sess.run(summary_scalar_val_op,feed_dict={self.accuracy_placeholder: mean_acc})
                self.writer.add_summary(summary_string,self.counter)
                self.save(self.saver,args.checkpoint_dir, self.counter)
            
            if self.counter % (30*self.batch_size) == 0 and self.counter!=0:
                summary_string1, summary_string2 = self.sess.run([summary_images_train_op,summary_images_val_op])
                self.writer.add_summary(summary_string1,self.counter)
                self.writer.add_summary(summary_string2,self.counter)
            
            if self.counter % (1*self.batch_size) == 0 and self.counter!=0:
                summary_string = self.sess.run(summary_scalar_train_op)
                self.writer.add_summary(summary_string,self.counter)
            
            if np.mod(self.counter, 100*self.batch_size) == 0 and self.counter !=0:
                self.save(self.saver,args.checkpoint_dir, self.counter)
            
            self.counter += self.batch_size
            
            total_time += time.time() - start_time
            time_left = (self.num_epochs*self.num_sample - self.counter)*total_time/(self.counter -start_iter)
            
            print(("Epoch: [%2d/%2d] [%4d/%4d] Loss: [%.4f] Time left: %s" \
                        % (self.counter//self.num_sample, self.num_epochs, self.counter%self.num_sample, self.num_sample, loss , datetime.timedelta(seconds=time_left)))) 

        coord.request_stop()
        coord.join(stop_grace_period_secs=10)

    def save(self, saver, checkpoint_dir, step):
        model_name = "U-Net"
        saver.save(self.sess,os.path.join(checkpoint_dir, model_name),global_step=step)

    def load(self, checkpoint_dir):
        def get_var_to_restore_list(ckpt_path, mask=[], prefix=""):
            """
            Get all the variable defined in a ckpt file and add them to the returned var_to_restore list. Allows for partially defined model to be restored fomr ckpt files.
            Args:
                ckpt_path: path to the ckpt model to be restored
                mask: list of layers to skip
                prefix: prefix string before the actual layer name in the graph definition
            """
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            variables_dict = {}
            for v in variables:
                name = v.name[:-2]
                skip=False
                #check for skip
                for m in mask:
                    if m in name:
                        skip=True
                        continue
                if not skip:
                    variables_dict[v.name[:-2]] = v
            #print(variables_dict)
            reader = tf.train.NewCheckpointReader(ckpt_path)
            var_to_shape_map = reader.get_variable_to_shape_map()
            var_to_restore = {}
            for key in var_to_shape_map:
                #print(key)
                if prefix+key in variables_dict.keys():
                    var_to_restore[key] = variables_dict[prefix+key]
            return var_to_restore

        print(" [*] Reading checkpoint...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            q = ckpt.model_checkpoint_path.split("-")[-1]
            print("Restored step: ", q)
            self.counter= int(q) 
            savvy = tf.train.Saver(var_list=get_var_to_restore_list(ckpt.model_checkpoint_path))
            savvy.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False

    def test(self, args):
        """Test""" 
        sample_op, sample_path,im_shape,sample_op_sem = self.build_input_image_op(args.input_list_val_test,is_test=True,num_epochs=1)               
        sample_op, sample_path,im_shape,sample_op_sem = tf.expand_dims(sample_op,axis=0), tf.expand_dims(sample_path,axis=0) , tf.expand_dims(im_shape,axis=0), tf.expand_dims(sample_op_sem,axis=0)
        
        sem_images = u_net_model(sample_op, self.options,name='u_net',is_test=True)
        sem_images_out = tf.argmax(sem_images, axis=-1, name="prediction")
        sem_images_out = tf.cast(tf.expand_dims(sem_images_out, axis=-1),tf.uint8)
        
        #init everything
        self.sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])

        #start queue runners
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners()
        print('Thread running')

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # write html for visual comparison
        if not os.path.exists(args.test_dir): #python 2 is dumb...
            os.makedirs(args.test_dir)

        print('Starting')
        count=0
        while count <= args.num_sample_test:
            print('Processed images: {}'.format(count), end='\r')
            pred_sem_imgs,sample_images,sample_paths,im_sps, sem_gt = self.sess.run([sem_images_out,sample_op,sample_path,im_shape,sample_op_sem])
            #iterate over each sample in the batch
            #create output destination
            print(sample_paths[0])
            dest_path = os.path.join(args.test_dir, sample_paths[0].decode('UTF-8').split("/")[-1])#sample_paths[rr].decode('UTF-8').replace("./testA",args.test_dir)
            parent_destination = os.path.abspath(os.path.join(dest_path, os.pardir))
            if not os.path.exists(parent_destination):
                os.makedirs(parent_destination)

            im_sp = im_sps[0]
            pred_sem_img = np.squeeze(pred_sem_imgs[0],axis=-1)
            misc.imsave(dest_path,pred_sem_img)
            
            count+=1

        print('Elaboration complete')
        coord.request_stop()
        coord.join(stop_grace_period_secs=30)
