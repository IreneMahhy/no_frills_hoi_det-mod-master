from utils.collections import AttrDict
import numpy as np

__C = AttrDict()

cfg = __C

__C.VCOCO_NUM_ACTION_CLASSES = 26

__C.VCOCO_NUM_TARGET_OBJECT_TYPES = 2

__C.VCOCO_ACTION_MASK = [[1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1],
                         [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

__C.VCOCO_NO_ROLE_ACTION_NUM = 5

__C.VCOCO_ACTION_NUM_WITH_ROLE = 29

__C.VCOCO_ACTION_NUM_REAL = 24

__C.action_classes = ['hold', 'sit', 'ride', 'look', 'hit', 'eat', 'jump', 'lay', 'talk_on_phone', 'carry',
                      'throw', 'catch', 'cut', 'work_on_computer', 'ski', 'surf', 'skateboard', 'drink',
                      'kick', 'read', 'snowboard', 'stand', 'walk', 'run', 'smile', 'point']

__C.action_no_obj = ['point', 'run', 'smile', 'stand', 'walk']

__C.action_one_obj = {'hold': 'obj', 'look': 'obj', 'carry': 'obj', 'throw': 'obj', 'catch': 'obj', 'kick': 'obj',
                      'read': 'obj',
                      'sit': 'instr', 'ride': 'instr', 'jump': 'instr', 'lay': 'instr', 'talk_on_phone': 'instr',
                      'work_on_computer': 'instr', 'ski': 'instr', 'surf': 'instr', 'skateboard': 'instr', 'drink': 'instr',
                      'point': 'instr', 'snowboard': 'instr'}

__C.action_roles = ['agent', 'instr', 'obj']


# object_prob_vecs = np.ones([num_cand, num_hois])
def apply_prior(Object_class, obj_prob):
    obj_prob[24:] = 0

    if Object_class != 32:  # not a snowboard, then the action is impossible to be snowboard
        obj_prob[23] = 0

    if Object_class != 74:  # not a book, then the action is impossible to be read
        obj_prob[22] = 0

    if Object_class != 33:  # not a sports ball, then the action is impossible to be kick
        obj_prob[21] = 0

    if (Object_class != 41) and (Object_class != 40) and (Object_class != 42) and (
            Object_class != 46):  # not 'wine glass', 'bottle', 'cup', 'bowl', then the action is impossible to be drink
        obj_prob[20] = 0

    if Object_class != 37:  # not a skateboard, then the action is impossible to be skateboard
        obj_prob[19] = 0

    if Object_class != 38:  # not a surfboard, then the action is impossible to be surf
        obj_prob[18] = 0

    if Object_class != 31:  # not a ski, then the action is impossible to be ski
        obj_prob[17] = 0

    if Object_class != 64:  # not a laptop, then the action is impossible to be work on computer
        obj_prob[16] = 0

    if (Object_class != 77) and (Object_class != 43) and (
            Object_class != 44):  # not 'scissors', 'fork', 'knife', then the action is impossible to be cut instr
        obj_prob[14] = 0

    if (Object_class != 33) and (
            Object_class != 30):  # not 'sports ball', 'frisbee', then the action is impossible to be throw and catch
        obj_prob[13] = 0
        obj_prob[12] = 0

    if Object_class != 68:  # not a cellphone, then the action is impossible to be talk_on_phone
        obj_prob[10] = 0

    if (Object_class != 14) and (Object_class != 61) and (Object_class != 62) and (Object_class != 60) and (
            Object_class != 58) and (
            Object_class != 57):  # not 'bench', 'dining table', 'toilet', 'bed', 'couch', 'chair', then the action is impossible to be lay
        obj_prob[9] = 0

    if (Object_class != 32) and (Object_class != 31) and (Object_class != 37) and (
            Object_class != 38):  # not 'snowboard', 'skis', 'skateboard', 'surfboard', then the action is impossible to be jump
        obj_prob[8] = 0

    if (Object_class != 47) and (Object_class != 48) and (Object_class != 49) and (Object_class != 50) and (
            Object_class != 51) and (Object_class != 52) and (Object_class != 53) and (Object_class != 54) and (
            Object_class != 55) and (
            Object_class != 56):  # not ''banana', 'apple', 'sandwich', 'orange', 'carrot', 'broccoli', 'hot dog', 'pizza', 'cake', 'donut', then the action is impossible to be eat_obj
        obj_prob[6] = 0

    if (Object_class != 43) and (Object_class != 44) and (
            Object_class != 45):  # not 'fork', 'knife', 'spoon', then the action is impossible to be eat_instr
        obj_prob[7] = 0

    if (Object_class != 39) and (
            Object_class != 35):  # not 'tennis racket', 'baseball bat', then the action is impossible to be hit_instr
        obj_prob[4] = 0

    if (Object_class != 33):  # not 'sports ball, then the action is impossible to be hit_obj
        obj_prob[5] = 0

    if (Object_class != 2) and (Object_class != 4) and (Object_class != 6) and (Object_class != 8) and (
            Object_class != 9) and (Object_class != 7) and (Object_class != 5) and (Object_class != 3) and (
            Object_class != 18) and (
            Object_class != 21):  # not 'bicycle', 'motorcycle', 'bus', 'truck', 'boat', 'train', 'airplane', 'car', 'horse', 'elephant', then the action is impossible to be ride
        obj_prob[2] = 0

    if (Object_class != 2) and (Object_class != 4) and (Object_class != 18) and (Object_class != 21) and (
            Object_class != 14) and (Object_class != 57) and (Object_class != 58) and (Object_class != 60) and (
            Object_class != 62) and (Object_class != 61) and (Object_class != 29) and (Object_class != 27) and (
            Object_class != 25):  # not 'bicycle', 'motorcycle', 'horse', 'elephant', 'bench', 'chair', 'couch', 'bed', 'toilet', 'dining table', 'suitcase', 'handbag', 'backpack', then the action is impossible to be sit
        obj_prob[1] = 0

    if (Object_class == -1):  # no object in pair, action is 'stand', 'walk', 'run' or 'smile'
        obj_prob[24:] = 1
        obj_prob[:24] = 0

    return obj_prob

def apply_prior_for_candidates(Object_class, verb_cls):

    if Object_class != 32:  # not a snowboard, then the action is impossible to be snowboard
        if verb_cls == 24:
            return False

    if Object_class != 74:  # not a book, then the action is impossible to be read
        if verb_cls == 23:
            return False

    if Object_class != 33:  # not a sports ball, then the action is impossible to be kick
        if verb_cls == 22:
            return False

    if (Object_class != 41) and (Object_class != 40) and (Object_class != 42) and (
            Object_class != 46):  # not 'wine glass', 'bottle', 'cup', 'bowl', then the action is impossible to be drink
        if verb_cls == 21:
            return False

    if Object_class != 37:  # not a skateboard, then the action is impossible to be skateboard
        if verb_cls == 20:
            return False

    if Object_class != 38:  # not a surfboard, then the action is impossible to be surf
        if verb_cls == 19:
            return False

    if Object_class != 31:  # not a ski, then the action is impossible to be ski
        if verb_cls == 18:
            return False

    if Object_class != 64:  # not a laptop, then the action is impossible to be work on computer
        if verb_cls == 17:
            return False

    if (Object_class != 77) and (Object_class != 43) and (
            Object_class != 44):  # not 'scissors', 'fork', 'knife', then the action is impossible to be cut instr
        if verb_cls == 15:
            return False

    if (Object_class != 33) and (
            Object_class != 30):  # not 'sports ball', 'frisbee', then the action is impossible to be throw and catch
        if verb_cls == 13 or verb_cls == 14:
            return False

    if Object_class != 68:  # not a cellphone, then the action is impossible to be talk_on_phone
        if verb_cls == 11:
            return False

    if (Object_class != 14) and (Object_class != 61) and (Object_class != 62) and (Object_class != 60) and (
            Object_class != 58) and (
            Object_class != 57):  # not 'bench', 'dining table', 'toilet', 'bed', 'couch', 'chair', then the action is impossible to be lay
        if verb_cls == 10:
            return False

    if (Object_class != 32) and (Object_class != 31) and (Object_class != 37) and (
            Object_class != 38):  # not 'snowboard', 'skis', 'skateboard', 'surfboard', then the action is impossible to be jump
        if verb_cls == 9:
            return False

    if (Object_class != 47) and (Object_class != 48) and (Object_class != 49) and (Object_class != 50) and (
            Object_class != 51) and (Object_class != 52) and (Object_class != 53) and (Object_class != 54) and (
            Object_class != 55) and (
            Object_class != 56):  # not ''banana', 'apple', 'sandwich', 'orange', 'carrot', 'broccoli', 'hot dog', 'pizza', 'cake', 'donut', then the action is impossible to be eat_obj
        if verb_cls == 7:
            return False

    if (Object_class != 43) and (Object_class != 44) and (
            Object_class != 45):  # not 'fork', 'knife', 'spoon', then the action is impossible to be eat_instr
        if verb_cls == 8:
            return False

    if (Object_class != 39) and (
            Object_class != 35):  # not 'tennis racket', 'baseball bat', then the action is impossible to be hit_instr
        if verb_cls == 5:
            return False

    if (Object_class != 33):  # not 'sports ball, then the action is impossible to be hit_obj
        if verb_cls == 6:
            return False

    if (Object_class != 2) and (Object_class != 4) and (Object_class != 6) and (Object_class != 8) and (
            Object_class != 9) and (Object_class != 7) and (Object_class != 5) and (Object_class != 3) and (
            Object_class != 18) and (
            Object_class != 21):  # not 'bicycle', 'motorcycle', 'bus', 'truck', 'boat', 'train', 'airplane', 'car', 'horse', 'elephant', then the action is impossible to be ride
        if verb_cls == 3:
            return False

    if (Object_class != 2) and (Object_class != 4) and (Object_class != 18) and (Object_class != 21) and (
            Object_class != 14) and (Object_class != 57) and (Object_class != 58) and (Object_class != 60) and (
            Object_class != 62) and (Object_class != 61) and (Object_class != 29) and (Object_class != 27) and (
            Object_class != 25):  # not 'bicycle', 'motorcycle', 'horse', 'elephant', 'bench', 'chair', 'couch', 'bed', 'toilet', 'dining table', 'suitcase', 'handbag', 'backpack', then the action is impossible to be sit
        if verb_cls == 2:
            return False

    return True
