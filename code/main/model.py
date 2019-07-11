import tensorflow as tf
import tensorflow.contrib.slim as slim
from functools import partial

from config import cfg
from tfflat.base import ModelDesc

from nets.basemodel import resnet50, resnet101, resnet152, resnet_arg_scope, resnet_v1
resnet_arg_scope = partial(resnet_arg_scope, bn_trainable=cfg.TRAIN.batch_norm)

class Model(ModelDesc):
     
    def head_net(self, blocks, is_training, trainable=True):
        
        normal_initializer = tf.truncated_normal_initializer(0, 0.01)
        msra_initializer = tf.contrib.layers.variance_scaling_initializer()
        xavier_initializer = tf.contrib.layers.xavier_initializer()
        
        with slim.arg_scope(resnet_arg_scope(bn_is_training=is_training)):
            
            out = slim.conv2d_transpose(blocks[-1], 256, [4, 4], stride=2,
                trainable=trainable, weights_initializer=normal_initializer,
                padding='SAME', activation_fn=tf.nn.relu,
                scope='up1')

            if(not cfg.MODEL.occluded_detection):

                out = slim.conv2d_transpose(out, 256, [4, 4], stride=2,
                                            trainable=trainable, weights_initializer=normal_initializer,
                                            padding='SAME', activation_fn=tf.nn.relu,
                                            scope='up2')

                out = slim.conv2d_transpose(out, 256, [4, 4], stride=2,
                    trainable=trainable, weights_initializer=normal_initializer,
                    padding='SAME', activation_fn=tf.nn.relu,
                    scope='up3')


                out = slim.conv2d(out, cfg.num_kps, [1,1],
                              trainable=trainable, weights_initializer=msra_initializer,
                              padding='SAME', normalizer_fn =None, activation_fn=None,
                              scope='out')
                return out
            else:
                if(not cfg.MODEL.occluded_cross_branch):
                    out = slim.conv2d_transpose(out, 256, [4, 4], stride=2,
                                                trainable=trainable, weights_initializer=normal_initializer,
                                                padding='SAME', activation_fn=tf.nn.relu,
                                                scope='up2')

                    out_occ = slim.conv2d_transpose(out, 256, [4, 4], stride=2,
                                                    trainable=trainable, weights_initializer=normal_initializer,
                                                    padding='SAME', activation_fn=tf.nn.relu,
                                                    scope='up3_occ')
                    out_vis = slim.conv2d_transpose(out, 256, [4, 4], stride=2,
                                                    trainable=trainable, weights_initializer=normal_initializer,
                                                    padding='SAME', activation_fn=tf.nn.relu,
                                                    scope='up3_vis')
                    out_vis = slim.conv2d(out_vis, cfg.num_kps + cfg.additional_outputs, [1, 1],
                                      trainable=trainable, weights_initializer=msra_initializer,
                                      padding='SAME', normalizer_fn=None, activation_fn=None,
                                      scope='out_vis')
                    out_occ = slim.conv2d(out_occ, cfg.num_kps + cfg.additional_outputs, [1, 1],
                                      trainable=trainable, weights_initializer=msra_initializer,
                                      padding='SAME', normalizer_fn=None, activation_fn=None,
                                      scope='out_occ')
                    return out_vis, out_occ
                else:

                    out_vis = slim.conv2d_transpose(out, 256, [4, 4], stride=2,
                                                trainable=trainable, weights_initializer=normal_initializer,
                                                padding='SAME', activation_fn=tf.nn.relu,
                                                scope='up2_vis')

                    out_occ = slim.conv2d_transpose(out, 256, [4, 4], stride=2,
                                                trainable=trainable , weights_initializer=normal_initializer,
                                                padding='SAME', activation_fn=tf.nn.relu,
                                                scope='up2_occ')

                    out_occ = slim.conv2d_transpose(out_occ, 256, [4, 4], stride=2,
                                                    trainable=trainable , weights_initializer=normal_initializer,
                                                    padding='SAME', activation_fn=tf.nn.relu,
                                                    scope='up3_occ')
                    out_vis = slim.conv2d_transpose(out_vis, 256, [4, 4], stride=2,
                                                    trainable=trainable, weights_initializer=normal_initializer,
                                                    padding='SAME', activation_fn=tf.nn.relu,
                                                    scope='up3_vis')


                    if(cfg.MODEL.stop_crossbranch_grad): #stop gradient for cross branching

                        out_vis_stack = tf.concat([tf.stop_gradient(out_occ), out_vis], axis=3)
                        out_occ_stack = tf.concat([out_occ, tf.stop_gradient(out_vis)], axis=3)

                    else:
                        out_vis_stack = tf.concat([out_occ, out_vis], axis=3)
                        out_occ_stack = tf.concat([out_occ, out_vis], axis=3)

                    out_vis = slim.conv2d(out_vis_stack, cfg.num_kps + cfg.additional_outputs, [1, 1],
                                          trainable=trainable, weights_initializer=msra_initializer,
                                          padding='SAME', normalizer_fn=None, activation_fn=None,
                                          scope='out_vis')
                    out_occ = slim.conv2d(out_occ_stack, cfg.num_kps + cfg.additional_outputs, [1, 1],
                                          trainable=trainable, weights_initializer=msra_initializer,
                                          padding='SAME', normalizer_fn=None, activation_fn=None,
                                          scope='out_occ')

                    return out_vis, out_occ




    def render_gaussian_heatmap(self, coord, output_shape, sigma):
        
        x = [i for i in range(output_shape[1])]
        y = [i for i in range(output_shape[0])]
        xx,yy = tf.meshgrid(x,y)
        xx = tf.reshape(tf.to_float(xx), (1,*output_shape,1))
        yy = tf.reshape(tf.to_float(yy), (1,*output_shape,1))
              
        x = tf.floor(tf.reshape(coord[:,:,0],[-1,1,1,cfg.num_kps]) / cfg.MODEL.input_shape[1] * output_shape[1] + 0.5)
        y = tf.floor(tf.reshape(coord[:,:,1],[-1,1,1,cfg.num_kps]) / cfg.MODEL.input_shape[0] * output_shape[0] + 0.5)

        heatmap = tf.exp(-(((xx-x)/tf.to_float(sigma))**2)/tf.to_float(2) -(((yy-y)/tf.to_float(sigma))**2)/tf.to_float(2))

        if (cfg.TRAIN.structure_aware_loss):
            for a, graph in enumerate(cfg.joint_graph):
                add = heatmap[:, :, :, graph[0]] + heatmap[:, :, :, graph[1]] + heatmap[:, :, :, graph[2]]
                add = tf.expand_dims(add, axis=3)
                heatmap = tf.concat([heatmap, add], axis=-1)


        return heatmap * 255.
   
    def make_network(self, is_train):
        if is_train:
            image = tf.placeholder(tf.float32, shape=[cfg.TRAIN.batch_size, *cfg.MODEL.input_shape, 3])
            target_coord = tf.placeholder(tf.float32, shape=[cfg.TRAIN.batch_size, cfg.num_kps, 2])
            valid = tf.placeholder(tf.float32, shape=[cfg.TRAIN.batch_size, cfg.num_kps])
            if(cfg.MODEL.interference_joints):
                interference_hm = (tf.placeholder(tf.float32, shape=[cfg.TRAIN.batch_size, *cfg.MODEL.output_shape, cfg.num_kps]))
                self.set_inputs(image, target_coord, valid, interference_hm)
            else:
                self.set_inputs(image, target_coord, valid)

        else:
            image = tf.placeholder(tf.float32, shape=[None, *cfg.MODEL.input_shape, 3])
            self.set_inputs(image)

        backbone = eval(cfg.MODEL.backbone)
        resnet_fms = backbone(image, is_train, bn_trainable=True)
        heatmap_outs = self.head_net(resnet_fms, is_train)

        
        if is_train:

            gt_heatmap = tf.stop_gradient(self.render_gaussian_heatmap(target_coord, cfg.MODEL.output_shape, cfg.sigma))
            if(cfg.MODEL.interference_joints):
                gt_heatmap = interference_hm + gt_heatmap
            valid_mask = tf.reshape(valid, [cfg.TRAIN.batch_size, 1, 1, cfg.num_kps])


            if(cfg.TRAIN.structure_aware_loss):
                for a, graph in enumerate(cfg.joint_graph):
                    add = heatmap_outs[:, :, :, graph[0]]  + heatmap_outs[:, :, :, graph[1]]  + heatmap_outs[:, :, :, graph[2]]
                    add = tf.expand_dims(add, axis=3)
                    if ( a== 0):
                        sal_hm = add
                    else:
                        sal_hm = tf.concat([sal_hm, add], axis=-1)
                s_loss = tf.reduce_mean(tf.square(heatmap_outs - gt_heatmap[:,:,:,:cfg.num_kps]) * valid_mask)
                sal_loss = tf.reduce_mean(tf.square(sal_hm - gt_heatmap[:,:,:,cfg.num_kps:])) * cfg.TRAIN.structure_aware_loss_weight
                tf.summary.scalar("sal_loss", sal_loss)
                tf.summary.scalar("single loss", s_loss)
                loss = sal_loss + s_loss
            else:
                loss = tf.reduce_mean(tf.square(heatmap_outs - gt_heatmap) * valid_mask)

            self.generate_summary_heatmaps(gt_heatmap[:,:,:,:cfg.num_kps], heatmap_outs, name="1_visible")
            tf.summary.image("0_input", image[0:1, ])
            tf.summary.scalar("total loss", loss)
            self.merged_summary = tf.summary.merge_all()
            self.add_tower_summary('loss', loss)
            self.set_loss(loss)
        else:
            self.set_outputs(heatmap_outs)

    def generate_summary_heatmaps(self, heatmap_pred, heatmap_gt, name):
        hm_list = (tf.unstack(heatmap_gt[0:1,], axis=3) +
                   tf.unstack(heatmap_pred[0:1,], axis=3))
        heatmaps = tf.stack(hm_list)
        heatmaps = tf.transpose(heatmaps, [0, 2, 3, 1])
        tf.summary.image(name,
            tf.contrib.gan.eval.image_grid(heatmaps,[2, cfg.num_kps], image_shape=cfg.MODEL.output_shape, num_channels =1))

    def make_occ_network(self, is_train):
        if is_train:
            image = tf.placeholder(tf.float32, shape=[cfg.TRAIN.batch_size, *cfg.MODEL.input_shape, 3])

            target_coord = tf.placeholder(tf.float32, shape=[cfg.TRAIN.batch_size, cfg.num_kps, 2])
            valid_vis = tf.placeholder(tf.float32, shape=[cfg.TRAIN.batch_size, cfg.num_kps])
            valid_occ = tf.placeholder(tf.float32, shape=[cfg.TRAIN.batch_size, cfg.num_kps])
            valid = tf.placeholder(tf.float32, shape=[cfg.TRAIN.batch_size, cfg.num_kps])
            self.set_inputs(image, target_coord, valid_vis, valid_occ, valid)
        else:
            image = tf.placeholder(tf.float32, shape=[None, *cfg.MODEL.input_shape, 3])
            self.set_inputs(image)

        backbone = eval(cfg.MODEL.backbone)
        resnet_fms = backbone(image, is_train, bn_trainable=True)

        heatmap_vis_pred, heatmap_occ_pred = self.head_net(resnet_fms, is_train)

        if is_train:

            gt_heatmap= tf.stop_gradient(self.render_gaussian_heatmap(target_coord, cfg.MODEL.output_shape, cfg.sigma))
            valid_mask_vis = tf.reshape(valid_vis, [cfg.TRAIN.batch_size, 1, 1, cfg.num_kps])

            valid_mask_occ = tf.reshape(valid_occ, [cfg.TRAIN.batch_size, 1, 1, cfg.num_kps])
            valid_mask = tf.reshape(valid, [cfg.TRAIN.batch_size, 1, 1, cfg.num_kps])

            if(cfg.MODEL.occluded_hard_loss):
                heatmap_occ_gt = gt_heatmap * valid_mask_occ
                heatmap_vis_gt = gt_heatmap * valid_mask_vis
                loss_occ = tf.reduce_mean(tf.square(heatmap_occ_pred - heatmap_occ_gt) * valid_mask)
                loss_vis = tf.reduce_mean(tf.square(heatmap_vis_pred - heatmap_vis_gt) * valid_mask)
                self.generate_summary_heatmaps(heatmap_occ_gt, heatmap_occ_pred, name="0_occluded")
                self.generate_summary_heatmaps(heatmap_vis_gt, heatmap_vis_pred, name="1_visible")
            else:
                loss_occ = tf.reduce_mean(tf.square(heatmap_occ_pred - gt_heatmap) * valid_mask_occ)
                loss_vis = tf.reduce_mean(tf.square(heatmap_vis_pred - gt_heatmap) * valid_mask_vis)
                self.generate_summary_heatmaps(gt_heatmap, heatmap_occ_pred, name="0_occluded")
                self.generate_summary_heatmaps(gt_heatmap, heatmap_vis_pred, name="1_visible")

            loss_sum = loss_vis + loss_occ * cfg.MODEL.occluded_loss_weight
            tf.summary.scalar("occluded loss", loss_occ)
            tf.summary.scalar("visible loss", loss_vis)
            tf.summary.scalar("total loss", loss_sum)
            tf.summary.image("3_input", image[0:1,])
            self.merged_summary = tf.summary.merge_all()
            self.add_tower_summary('loss', loss_sum)
            self.set_loss(loss_sum)
        else:
            self.set_outputs(heatmap_vis_pred, heatmap_occ_pred)