import os
import h5py
import copy
import itertools
import numpy as np
from torch.utils.data import Dataset

import utils.io as io
from utils.constants import Constants
from data.coco_classes import COCO_CLASSES
from data.hico.hico_constants import HicoConstants
from data.vcoco.vcoco_constants import VcocoConstants
from data.vcoco_cfg import cfg
from data.vcoco_cfg import apply_prior


class FeatureConstantsVcoco(VcocoConstants, io.JsonSerializableClass):
    def __init__(
            self,
            subset,
            clean_dir=os.path.join(os.getcwd(), 'data_symlinks/vcoco_clean'),
            proc_dir=os.path.join(os.getcwd(), 'data_symlinks/vcoco_processed'),
            hoi_cand_dir=os.path.join(
                os.getcwd(),
                'data_symlinks/vcoco_exp/hoi_candidates')):
        super(FeatureConstantsVcoco, self).__init__(
            clean_dir=clean_dir,
            proc_dir=proc_dir)
        self.subset = subset
        self.hoi_cand_dir = hoi_cand_dir
        self.hoi_cands_hdf5 = os.path.join(
            self.hoi_cand_dir,
            f'hoi_candidates_{subset}.hdf5')
        self.hoi_cand_labels_hdf5 = os.path.join(
            self.hoi_cand_dir,
            f'hoi_candidate_labels_{subset}.hdf5')
        self.faster_rcnn_feats_hdf5 = os.path.join(
            self.proc_dir,
            'faster_rcnn_fc7.hdf5')
        self.box_feats_hdf5 = os.path.join(
            self.hoi_cand_dir,
            f'hoi_candidates_box_feats_{subset}.hdf5')
        self.human_pose_feats_hdf5 = os.path.join(
            self.hoi_cand_dir,
            f'human_pose_feats_{subset}.hdf5')
        self.num_pose_keypoints = 18
        self.balanced_sampling = True
        self.fp_to_tp_ratio = 1000
        self.all_object_class_scores = False


class FeatureConstantsHico(HicoConstants, io.JsonSerializableClass):
    def __init__(
            self,
            subset,
            clean_dir=os.path.join(os.getcwd(), 'data_symlinks/hico_clean'),
            proc_dir=os.path.join(os.getcwd(), 'data_symlinks/hico_processed'),
            hoi_cand_dir=os.path.join(
                os.getcwd(),
                'data_symlinks/hico_exp/hoi_candidates')):
        super(FeatureConstantsHico, self).__init__(
            clean_dir=clean_dir,
            proc_dir=proc_dir)
        self.subset = subset
        self.hoi_cand_dir = hoi_cand_dir
        self.hoi_cands_hdf5 = os.path.join(  # box pair candidates
            self.hoi_cand_dir,
            f'hoi_candidates_{subset}.hdf5')
        self.hoi_cand_labels_hdf5 = os.path.join(
            self.hoi_cand_dir,
            f'hoi_candidate_labels_{subset}.hdf5')
        self.faster_rcnn_feats_hdf5 = os.path.join(
            self.proc_dir,
            'faster_rcnn_fc7.hdf5')
        self.box_feats_hdf5 = os.path.join(
            self.hoi_cand_dir,
            f'hoi_candidates_box_feats_{subset}.hdf5')
        self.human_pose_feats_hdf5 = os.path.join(
            self.hoi_cand_dir,
            f'human_pose_feats_{subset}.hdf5')
        self.num_pose_keypoints = 18
        self.balanced_sampling = True
        self.fp_to_tp_ratio = 1000
        self.all_object_class_scores = False


class Features(Dataset):
    def __init__(self, const):
        self.const = copy.deepcopy(const)
        self.hoi_cands = self.load_hdf5_file(self.const.hoi_cands_hdf5)  # load box pair candidates
        self.hoi_cand_labels = self.load_hdf5_file(
            self.const.hoi_cand_labels_hdf5)
        self.faster_rcnn_feats = self.load_hdf5_file(
            self.const.faster_rcnn_feats_hdf5)
        self.global_ids = self.load_subset_ids(self.const.subset)
        self.hoi_dict = self.get_hoi_dict(self.const.hoi_list_json)
        self.obj_to_hoi_ids = self.get_obj_to_hoi_ids(self.hoi_dict)
        self.obj_to_id = self.get_obj_to_id(self.const.object_list_json)
        self.verb_to_id = self.get_verb_to_id(self.const.verb_list_json)
        self.anno_dict = self.get_anno_dict(self.const.anno_list_json)
        self.corre_mat = np.load(self.const.mat_npy)
        self.obj_to_coco_id = {
            k: v for k, v in zip(COCO_CLASSES, range(len(COCO_CLASSES)))}
        if self.const.box_feats_hdf5:
            self.box_feats = self.load_hdf5_file(self.const.box_feats_hdf5)
        if self.const.human_pose_feats_hdf5:
            self.human_pose_feat = self.load_hdf5_file(
                self.const.human_pose_feats_hdf5)

    def get_anno_dict(self, anno_list_json):
        anno_list = io.load_json_object(anno_list_json)
        anno_dict = {anno['global_id']: anno for anno in anno_list}
        return anno_dict

    def load_hdf5_file(self, hdf5_filename, mode='r'):
        return h5py.File(hdf5_filename, mode)

    def get_hoi_dict(self, hoi_list_json):
        hoi_list = io.load_json_object(hoi_list_json)
        hoi_dict = {hoi['id']: hoi for hoi in hoi_list}
        return hoi_dict

    def get_obj_to_id(self, object_list_json):
        object_list = io.load_json_object(object_list_json)
        obj_to_id = {obj['name']: obj['id'] for obj in object_list}
        return obj_to_id

    def get_verb_to_id(self, verb_list_json):
        verb_list = io.load_json_object(verb_list_json)
        verb_to_id = {verb['name']: verb['id'] for verb in verb_list}
        return verb_to_id

    def get_obj_to_hoi_ids(self, hoi_dict):
        obj_to_hoi_ids = {}
        for hoi_id, hoi in hoi_dict.items():
            obj = hoi['object']
            if obj in obj_to_hoi_ids:
                obj_to_hoi_ids[obj].append(hoi_id)
            else:
                obj_to_hoi_ids[obj] = [hoi_id]

        return obj_to_hoi_ids

    def load_subset_ids(self, subset):
        split_ids = io.load_json_object(self.const.split_ids_json)
        return sorted(split_ids[subset])

    def __len__(self):
        return len(self.global_ids)

    def get_labels(self, global_id):
        # hoi_idx: number in [0,num]
        hoi_cands = self.hoi_cands[global_id]
        hoi_idxs = hoi_cands['boxes_scores_rpn_ids_hoi_idx'][:, -1]
        hoi_idxs = hoi_idxs.astype(np.int)
        # label: 0/1 indicating if there was a match with any gt for that hoi
        labels = self.hoi_cand_labels[global_id][()]
        num_cand = labels.shape[0]
        hoi_label_vecs = np.zeros([num_cand, len(self.hoi_dict)])
        hoi_label_vecs[np.arange(num_cand), hoi_idxs] = labels

        hoi_ids = [None]*num_cand
        for i in range(num_cand):
            hoi_ids[i] = str(hoi_idxs[i]+1).zfill(3)

        return hoi_ids, labels, hoi_label_vecs


    # 将human和object score填充到(N*hoi_class_num)矩阵中所有对应的位置
    def get_faster_rcnn_prob_vecs(self, hoi_ids, human_probs, object_probs, object_cls_ids):
        num_hois = len(self.hoi_dict)
        num_cand = len(hoi_ids)
        human_prob_vecs = np.tile(np.expand_dims(human_probs, 1), [1, 24])

        if isinstance(self.const, FeatureConstantsVcoco):  # 对相关动作类填入object_score，no_role_action分数为1
            object_prob_vecs = np.ones([num_cand, 24])
            '''
            for i in range(num_cand):
                obj_cls = object_cls_ids[i] - 1
                verb_cls = np.where(self.corre_mat[:, obj_cls] == 1)[0]
                object_prob_vecs[i, verb_cls] = object_probs[i]
            
            for i, hoi_id in enumerate(hoi_ids):
                if not self.hoi_dict[hoi_id]['object']:  # 候选对样本对应的类没有role
                    object_prob_vecs[i, :-5] = 0
                    return human_prob_vecs, object_prob_vecs

                object_prob_vecs[i, :-5] = object_probs[i]
                object_prob_vecs[i, -5:] = 0
            '''
            for i in range(num_cand):
                object_prob_vecs[i] = apply_prior(object_cls_ids[i], object_prob_vecs[i])
                if object_probs[i] != 0:
                    object_prob_vecs[i] *= object_probs[i]
        else:
            object_prob_vecs = np.zeros([num_cand, num_hois])
            for i, hoi_id in enumerate(hoi_ids):  # hoi_id指的不是pair的gt，而是表明这是该hoi的候选pair
                obj = self.hoi_dict[hoi_id]['object']
                obj_hoi_ids = self.obj_to_hoi_ids[obj]
                for obj_hoi_id in obj_hoi_ids:
                    object_prob_vecs[i, int(obj_hoi_id) - 1] = object_probs[i]

        return human_prob_vecs, object_prob_vecs

    def sample_cands(self, hoi_labels):
        num_cands = hoi_labels.shape[0]
        indices = np.arange(num_cands)
        tp_ids = indices[hoi_labels == 1.0]
        fp_ids = indices[hoi_labels == 0]
        num_tp = tp_ids.shape[0]
        num_fp = fp_ids.shape[0]
        if num_tp == 0:
            num_fp_to_sample = self.const.fp_to_tp_ratio
        else:
            num_fp_to_sample = min(num_fp, self.const.fp_to_tp_ratio*num_tp)
        sampled_fp_ids = np.random.permutation(fp_ids)[:num_fp_to_sample]
        sampled_ids = np.concatenate((tp_ids, sampled_fp_ids), 0)
        return sampled_ids

    def get_obj_one_hot_coco(self, hoi_ids):
        num_cand = len(hoi_ids)
        obj_one_hot = np.zeros([num_cand, len(self.obj_to_id)])

        for i, hoi_id in enumerate(hoi_ids):
            obj_id = self.obj_to_id[self.hoi_dict[hoi_id]['object']]
            obj_idx = int(obj_id)-1
            obj_one_hot[i, obj_idx] = 1.0

        return obj_one_hot

    def get_obj_one_hot_vcoco(self, obj_ids):
        num_cand = len(obj_ids)
        obj_one_hot = np.zeros([num_cand, len(self.obj_to_id)])

        for i, obj_id in enumerate(obj_ids):
            obj_idx = int(obj_id)-1
            obj_one_hot[i, obj_idx] = 1.0

        return obj_one_hot

    def get_verb_one_hot(self, hoi_ids):
         num_cand = len(hoi_ids)
         verb_one_hot = np.zeros([num_cand, len(self.verb_to_id)])
         for i, hoi_id in enumerate(hoi_ids):
             verb_id = self.verb_to_id[self.hoi_dict[hoi_id]['verb']]
             verb_idx = int(verb_id)-1
             verb_one_hot[i, verb_idx] = 1.0
         return verb_one_hot

    def get_prob_mask(self, hoi_idx):
        num_cand = len(hoi_idx)
        prob_mask = np.zeros([num_cand, len(self.hoi_dict)])
        prob_mask[np.arange(num_cand), hoi_idx] = 1.0
        return prob_mask

    def get_im_wh(self, global_id, num_cand):
        h, w = self.anno_dict[global_id]['image_size'][:2]
        im_wh = np.ones([num_cand, 2], dtype=np.float32)
        im_wh[:, 0] = im_wh[:, 0]*w
        im_wh[:, 1] = im_wh[:, 1]*h
        return im_wh

    def get_obj_prob_vec(
            self,
            global_id,
            object_rpn_id):
        scores_npy = os.path.join(
            self.const.faster_rcnn_boxes,
            f'{global_id}_scores.npy')
        scores = np.load(scores_npy)
        scores = scores[object_rpn_id, :]
        num_hois = len(self.hoi_dict)
        gather_ids = np.zeros([num_hois], dtype=np.int)
        for obj, obj_hoi_ids in self.obj_to_hoi_ids.items():
            obj_idx = self.obj_to_coco_id[' '.join(obj.split('_'))]
            #obj_idx = int(self.obj_to_id[obj])-1
            obj_hoi_idx = [int(v)-1 for v in obj_hoi_ids]
            gather_ids[obj_hoi_idx] = obj_idx  # 得到每个hoi类别对应的物体在COCO中对应的id
        obj_prob_vec = scores[:, gather_ids]
        return obj_prob_vec

    def get_unique_pair(self, human_rpn, object_rpn, human_box, object_box):
        human_index = []
        human_uni_list = []
        for i in range(human_rpn.size):
            if human_rpn[i] not in human_uni_list:
                human_index.append(i)
                human_uni_list.append(human_rpn[i])

        object_index = []
        object_uni_list = []
        for i in range(object_rpn.size):
            if object_rpn[i] not in object_uni_list:
                object_index.append(i)
                object_uni_list.append(object_rpn[i])

        unique_index = []
        for i in human_uni_list:
            for j in object_uni_list:
                for x in range(human_rpn.shape[0]):
                    if human_rpn[x] == i and object_rpn[x] == j:
                        unique_index.append(x)
                        break

        human_box_arr = human_box[np.array(human_index)]
        object_box_arr = object_box[np.array(object_index)]
        '''
        human_box_tuple = list(human_box)
        for i, box in enumerate(human_box_tuple):
            human_box_tuple[i] = tuple(box)
        human_box_list = list(set(human_box_tuple))
        human_box_list.sort(key=human_box_tuple.index)
        for i, box in enumerate(human_box_list):
            human_box_list[i] = np.array(box)

        object_box_tuple = list(object_box)
        for i, box in enumerate(object_box_tuple):
            object_box_tuple[i] = tuple(box)
        object_box_list = list(set(object_box_tuple))
        if (0, 0, 0, 0) in object_box_list:
            object_box_list.remove((0, 0, 0, 0))
        object_box_list.sort(key=object_box_tuple.index)
        for i, box in enumerate(object_box_list):
            object_box_list[i] = np.array(box)
        '''
        return np.array(unique_index), human_box_arr, object_box_arr

    '''
    def get_no_role_pair(self, human_rpn, object_rpn):
        index = list()
        human_set = set(list(human_rpn))
        for i in human_set:
            for x in range(human_rpn.shape[0]):
                if human_rpn[x] == i and object_rpn[x] == -1:
                    index.append(x)
                    break

        return np.array(index)
    '''

    def __getitem__(self, i):
        global_id = self.global_ids[i]

        start_end_ids = self.hoi_cands[global_id]['start_end_ids'][()]
        hoi_cands_ = self.hoi_cands[global_id]['boxes_scores_rpn_ids_hoi_idx'][()]
        # hoi_id标示ho pair是作为哪类的候选对，labels标示该pair对比gt是否正样本（0或1）
        hoi_ids_, hoi_labels_, hoi_label_vecs_ = self.get_labels(global_id)
        if self.const.box_feats_hdf5:
            box_feats_ = self.box_feats[global_id][()]
        else:
            box_feats_ = None

        if self.const.human_pose_feats_hdf5:
            absolute_pose_feat_ = self.human_pose_feat[global_id]['absolute_pose'][()]
            relative_pose_feat_ = self.human_pose_feat[global_id]['relative_pose'][()]
        else:
            absolute_pose_feat_ = None
            relative_pose_feat_ = None

        if self.const.balanced_sampling:
            cand_ids = self.sample_cands(hoi_labels_)
            hoi_cands = hoi_cands_[cand_ids]
            hoi_ids = np.array(hoi_ids_)[cand_ids].tolist()
            hoi_labels = hoi_labels_[cand_ids]
            hoi_label_vecs = hoi_label_vecs_[cand_ids]
            if box_feats_ is not None:
                box_feats = box_feats_[cand_ids]
            else:
                box_feats = None
            if absolute_pose_feat_ is not None:
                absolute_pose_feat = absolute_pose_feat_[cand_ids]
                relative_pose_feat = relative_pose_feat_[cand_ids]
        else:
            hoi_cands = hoi_cands_
            hoi_ids = hoi_ids_
            hoi_labels = hoi_labels_
            hoi_label_vecs = hoi_label_vecs_
            box_feats = box_feats_
            absolute_pose_feat = absolute_pose_feat_
            relative_pose_feat = relative_pose_feat_

        to_return = {
            'global_id': global_id,
            'human_box': hoi_cands[:, :4],
            'object_box': hoi_cands[:, 4:8],
            'human_prob': hoi_cands[:, 8],  # human score
            'object_prob': hoi_cands[:, 9],
            'human_rpn_id': hoi_cands[:, 10].astype(np.int),
            'object_rpn_id': hoi_cands[:, 11].astype(np.int),
            'object_cls_id': hoi_cands[:, 12].astype(np.int),  # 1~80
            'hoi_idx': hoi_cands[:, -1].astype(np.int),  # conversion of hoi_id to 0 indexed integers
            'hoi_id': hoi_ids,  # 1 based index
            'hoi_label': hoi_labels,
            'hoi_label_vec': hoi_label_vecs,  # 将每个候选对的hoi_label标示扩展到N个动作类别上
            'box_feat': box_feats,
            'absolute_pose': absolute_pose_feat,
            'relative_pose': relative_pose_feat,
            'hoi_cands_': hoi_cands_,
            'start_end_ids_': start_end_ids.astype(np.int),  # Corresponds to non sampled hoi_cands_ which is the same as hoi_cands when balanced sampling is not used
        }

        to_return['human_feat'] = np.take(
            self.faster_rcnn_feats[global_id],
            to_return['human_rpn_id'],
            axis=0)
        to_return['object_feat'] = np.take(
            self.faster_rcnn_feats[global_id],
            to_return['object_rpn_id'],
            axis=0)

        human_prob_vecs, object_prob_vecs = self.get_faster_rcnn_prob_vecs(
            to_return['hoi_id'],
            to_return['human_prob'],
            to_return['object_prob'],
            to_return['object_cls_id'])
        if self.const.all_object_class_scores is True:
            object_prob_vecs = self.get_obj_prob_vec(
                global_id,
                to_return['object_rpn_id'])
        to_return['human_prob_vec'] = human_prob_vecs
        to_return['object_prob_vec'] = object_prob_vecs

        if isinstance(self.const, FeatureConstantsVcoco):
            to_return['object_one_hot'] = self.get_obj_one_hot_vcoco(to_return['object_cls_id'])
            to_return['verb_one_hot'] = hoi_label_vecs
        else:
            to_return['object_one_hot'] = self.get_obj_one_hot_coco(to_return['hoi_id'])
            to_return['verb_one_hot'] = self.get_verb_one_hot(to_return['hoi_id'])

        to_return['prob_mask'] = self.get_prob_mask(to_return['hoi_idx'])  # 每张图中所有ho pair对所有类别的矩阵，类别存在为1
        # 同一对ho唯一的编号，以及不重复的human_num和object_num
        to_return['unique_index'], to_return['human_boxes'], to_return['object_boxes'] = \
            self.get_unique_pair(to_return['human_rpn_id'], to_return['object_rpn_id'],
                                 to_return['human_box'], to_return['object_box'])

        return to_return
