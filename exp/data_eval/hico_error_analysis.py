import os
import argparse
import time
import h5py
from tqdm import tqdm
import numpy as np
# import matplotlib.pyplot as plt
from multiprocessing import Pool
from sklearn.metrics import average_precision_score, precision_recall_curve

import utils.io as io
from utils.bbox_utils import compute_iou

parser = argparse.ArgumentParser()
parser.add_argument(
    '--pred_hoi_dets_hdf5',
    type=str,
    default=None,
    required=True,
    help='Path to predicted hoi detections hdf5 file')
parser.add_argument(
    '--out_dir',
    type=str,
    default=None,
    required=True,
    help='Output directory')
parser.add_argument(
    '--proc_dir',
    type=str,
    default=None,
    required=True,
    help='Path to processed data directory')
parser.add_argument(
    '--subset',
    type=str,
    default='test',
    choices=['train', 'test', 'val', 'train_val'],
    help='Subset of data to run the evaluation on')
parser.add_argument(
    '--num_processes',
    type=int,
    default=12,
    help='Number of processes to parallelize across')


def match_hoi(pred_det, gt_dets, detected):
    flag2 = 1  # bck
    flag3 = 1  # human misloc
    flag4 = 1  # obj misloc
    flag5 = 0  # duplicate detection
    flag6 = 1  # mis grouping
    is_match = False
    remaining_gt_dets = [gt_det for gt_det in gt_dets]
    for i, gt_det in enumerate(gt_dets):
        human_iou = compute_iou(pred_det['human_box'], gt_det['human_box'])
        if human_iou > 0.1:
            flag2 = 0
        if human_iou > 0.5:
            flag3 = 0
            object_iou = compute_iou(pred_det['object_box'], gt_det['object_box'])
            if object_iou > 0.1:
                flag6 = 0
            if object_iou > 0.5:
                flag4 = 0
                is_match = True
                detected.append(gt_det)
                del remaining_gt_dets[i]
                break
        # remaining_gt_dets.append(gt_det)

    if not is_match:
        for i, gt_det in enumerate(detected):
            human_iou = compute_iou(pred_det['human_box'], gt_det['human_box'])
            if human_iou > 0.5:
                object_iou = compute_iou(pred_det['object_box'], gt_det['object_box'])
                if object_iou > 0.5:
                    flag2, flag3, flag4, flag6 = 0, 0, 0, 0
                    flag5 = 1
                    break

    return is_match, remaining_gt_dets, detected, (flag2, flag3, flag4, flag5, flag6)


def compute_ap(precision, recall):
    if np.any(np.isnan(recall)):
        return np.nan

    ap = 0
    for t in np.arange(0, 1.1, 0.1):  # 0, 0.1, 0.2, ..., 1.0
        selected_p = precision[recall >= t]
        if selected_p.size == 0:
            p = 0
        else:
            p = np.max(selected_p)
        ap += p / 11.

    return ap


def compute_pr(y_true, y_score, npos):
    sorted_y_true = [y for y, _ in
                     sorted(zip(y_true, y_score), key=lambda x: x[1], reverse=True)]
    tp = np.array(sorted_y_true)
    fp = ~tp
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    if npos == 0:
        recall = np.nan * tp
    else:
        recall = tp / npos
    precision = tp / (tp + fp)
    return precision, recall


def compute_normalized_pr(y_true, y_score, npos, N=196.45):
    sorted_y_true = [y for y, _ in
                     sorted(zip(y_true, y_score), key=lambda x: x[1], reverse=True)]
    tp = np.array(sorted_y_true)
    fp = ~tp
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    if npos == 0:
        recall = np.nan * tp
    else:
        recall = tp / npos
    precision = recall * N / (recall * N + fp)
    nap = np.sum(precision[sorted_y_true]) / (npos + 1e-6)
    return precision, recall, nap


def eval_hoi(hoi_id, global_ids, gt_dets, pred_dets_hdf5, out_dir):
    print(f'Evaluating hoi_id: {hoi_id} ...')
    pred_dets = h5py.File(pred_dets_hdf5, 'r')

    fp1 = []  # incorrect label
    fp2 = []  # bck
    fp3 = []  # person misloc
    fp4 = []  # obj misloc
    fp5 = []  # duplicate detection
    fp6 = []  # mis-grouping
    fp7 = []  # occlusion

    y_true = []
    y_score = []
    det_id = []
    ndet = 0
    npos = 0
    for global_id in global_ids:
        flag1 = 0
        if hoi_id in gt_dets[global_id]:
            candidate_gt_dets = gt_dets[global_id][hoi_id]
        else:
            candidate_gt_dets = []
            flag1 = 1
        npos += len(candidate_gt_dets)

        start_id, end_id = pred_dets[global_id]['start_end_ids'][int(hoi_id) - 1]
        hoi_dets = \
            pred_dets[global_id]['human_obj_boxes_scores'][start_id:end_id]

        num_dets = hoi_dets.shape[0]
        sorted_idx = [idx for idx, _ in sorted(
            zip(range(num_dets), hoi_dets[:, 8].tolist()),
            key=lambda x: x[1],
            reverse=True)]
        detected = []  # 已经匹配过的废弃gt，判断是否重复检测
        for i in sorted_idx:
            ndet += 1
            pred_det = {
                'human_box': hoi_dets[i, :4],
                'object_box': hoi_dets[i, 4:8],
                'score': hoi_dets[i, 8]
            }
            is_match, candidate_gt_dets, detected, flags = match_hoi(pred_det, candidate_gt_dets, detected)
            y_true.append(is_match)
            y_score.append(pred_det['score'])
            det_id.append((global_id, i))

            flag2, flag3, flag4, flag5, flag6 = flags
            if not is_match:
                if flag2:  # bck
                    fp1.append(0)
                    fp2.append(1)
                    fp3.append(0)
                    fp4.append(0)
                    fp5.append(0)
                    fp6.append(0)
                    fp7.append(0)
                elif flag3:  # person misloc
                    fp1.append(0)
                    fp2.append(0)
                    fp3.append(1)
                    fp4.append(0)
                    fp5.append(0)
                    fp6.append(0)
                    fp7.append(0)
                elif flag1:
                    fp1.append(1)
                    fp2.append(0)
                    fp3.append(0)
                    fp4.append(0)
                    fp5.append(0)
                    fp6.append(0)
                    fp7.append(0)
                elif flag6:
                    fp1.append(0)
                    fp2.append(0)
                    fp3.append(0)
                    fp4.append(0)
                    fp5.append(0)
                    fp6.append(6)
                    fp7.append(0)
                elif flag4:
                    fp1.append(0)
                    fp2.append(0)
                    fp3.append(0)
                    fp4.append(1)
                    fp5.append(0)
                    fp6.append(0)
                    fp7.append(0)
                elif flag5:
                    fp1.append(0)
                    fp2.append(0)
                    fp3.append(0)
                    fp4.append(0)
                    fp5.append(1)
                    fp6.append(0)
                    fp7.append(0)
            else:
                fp1.append(0)
                fp2.append(0)
                fp3.append(0)
                fp4.append(0)
                fp5.append(0)
                fp6.append(0)
                fp7.append(0)

    a_fp1 = np.array(fp1, dtype=np.float32)
    a_fp2 = np.array(fp2, dtype=np.float32)
    a_fp3 = np.array(fp3, dtype=np.float32)
    a_fp4 = np.array(fp4, dtype=np.float32)
    a_fp5 = np.array(fp5, dtype=np.float32)
    a_fp6 = np.array(fp6, dtype=np.float32)
    a_fp7 = np.array(fp7, dtype=np.float32)
    a_sc = np.array(y_score, dtype=np.float32)
    a_tp = np.array(y_true, dtype=np.float32)

    idx = a_sc.argsort()[::-1]
    a_fp1 = a_fp1[idx]
    a_fp2 = a_fp2[idx]
    a_fp3 = a_fp3[idx]
    a_fp4 = a_fp4[idx]
    a_fp5 = a_fp5[idx]
    a_fp6 = a_fp6[idx]
    a_fp7 = a_fp7[idx]
    a_tp = a_tp[idx]
    a_sc = a_sc[idx]

    num_inst = int(min(npos, len(a_sc)))
    a_fp1 = a_fp1[:num_inst]
    a_fp2 = a_fp2[:num_inst]
    a_fp3 = a_fp3[:num_inst]
    a_fp4 = a_fp4[:num_inst]
    a_fp5 = a_fp5[:num_inst]
    a_fp6 = a_fp6[:num_inst]
    a_fp7 = a_fp7[:num_inst]
    a_tp = a_tp[:num_inst]
    a_sc = a_sc[:num_inst]
    frac_fp1 = np.sum(a_fp1) / (num_inst - np.sum(a_tp))
    frac_fp2 = np.sum(a_fp2) / (num_inst - np.sum(a_tp))
    frac_fp3 = np.sum(a_fp3) / (num_inst - np.sum(a_tp))
    frac_fp4 = np.sum(a_fp4) / (num_inst - np.sum(a_tp))
    frac_fp5 = np.sum(a_fp5) / (num_inst - np.sum(a_tp))
    frac_fp6 = np.sum(a_fp6) / (num_inst - np.sum(a_tp))
    frac_fp7 = np.sum(a_fp7) / (num_inst - np.sum(a_tp))

    errors = {
        'fp_inc': frac_fp1,
        'fp_bck': frac_fp2,
        'fp_Hmis': frac_fp3,
        'fp_Omis': frac_fp4,
        'fp_dupl': frac_fp5,
        'fp_misg': frac_fp6,
        'fp_occl': frac_fp7,
        'tp': np.sum(a_tp),
        'rec': np.sum(a_tp) / float(npos),
        'prec': np.sum(a_tp) / np.maximum(
            np.sum(a_fp1) + np.sum(a_fp2) + np.sum(a_fp3) + np.sum(a_fp4) + np.sum(a_fp5) + np.sum(
                a_fp6) + np.sum(a_fp7) + np.sum(a_tp), np.finfo(np.float64).eps),
        'ndet': ndet,
        'npos': npos
    }
    '''
    hoi_idx = int(hoi_id)-1
    fp_inc[hoi_idx] = frac_fp1
    fp_bck[hoi_idx] = frac_fp2
    fp_Hmis[hoi_idx] = frac_fp3
    fp_Omis[hoi_idx] = frac_fp4
    fp_dupl[hoi_idx] = frac_fp5
    fp_misg[hoi_idx] = frac_fp6
    fp_occl[hoi_idx] = frac_fp7
    tp_[hoi_idx] = np.sum(a_tp)
    rec[hoi_idx] = np.sum(a_tp) / float(npos)
    prec[hoi_idx] = np.sum(a_tp) / np.maximum(
        np.sum(a_fp1) + np.sum(a_fp2) + np.sum(a_fp3) + np.sum(a_fp4) + np.sum(a_fp5) + np.sum(
            a_fp6) + np.sum(a_fp7) + np.sum(a_tp), np.finfo(np.float64).eps)
    npos_[hoi_idx] = npos

    error_data['fp_inc'] = fp_inc
    error_data['fp_bck'] = fp_bck
    error_data['fp_Hmis'] = fp_Hmis
    error_data['fp_Omis'] = fp_Omis
    error_data['fp_dupl'] = fp_dupl
    error_data['fp_misg'] = fp_misg
    error_data['fp_occl'] = fp_occl
    error_data['tp_'] = tp_
    error_data['rec'] = rec
    error_data['prec'] = prec
    error_data['ndet'] = ndet
    error_data['npos'] = npos_
    '''

    # Compute PR
    precision, recall = compute_pr(y_true, y_score, npos)
    # nprecision,nrecall,nap = compute_normalized_pr(y_true,y_score,npos)

    # Compute AP
    ap = compute_ap(precision, recall)
    print(f'AP:{ap}')

    # Plot PR curve
    # plt.figure()
    # plt.step(recall,precision,color='b',alpha=0.2,where='post')
    # plt.fill_between(recall,precision,step='post',alpha=0.2,color='b')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('Precision-Recall curve: AP={0:0.4f}'.format(ap))
    # plt.savefig(
    #     os.path.join(out_dir,f'{hoi_id}_pr.png'),
    #     bbox_inches='tight')
    # plt.close()

    # Save AP data
    ap_data = {
        'y_true': y_true,
        'y_score': y_score,
        'det_id': det_id,
        'npos': npos,
        'ap': ap,
    }
    np.save(
        os.path.join(out_dir, f'{hoi_id}_ap_data.npy'),
        ap_data)

    return (ap, hoi_id, errors)


def load_gt_dets(proc_dir, global_ids_set):
    # Load anno_list
    print('Loading anno_list.json ...')
    anno_list_json = os.path.join(proc_dir, 'anno_list.json')
    anno_list = io.load_json_object(anno_list_json)

    gt_dets = {}
    for anno in anno_list:
        if anno['global_id'] not in global_ids_set:
            continue

        global_id = anno['global_id']
        gt_dets[global_id] = {}
        for hoi in anno['hois']:
            hoi_id = hoi['id']
            gt_dets[global_id][hoi_id] = []
            for human_box_num, object_box_num in hoi['connections']:
                human_box = hoi['human_bboxes'][human_box_num]
                object_box = hoi['object_bboxes'][object_box_num]
                det = {
                    'human_box': human_box,
                    'object_box': object_box,
                }
                gt_dets[global_id][hoi_id].append(det)

    return gt_dets


def main():
    args = parser.parse_args()

    print('Creating output dir ...')
    io.mkdir_if_not_exists(args.out_dir, recursive=True)

    # Load hoi_list
    hoi_list_json = os.path.join(args.proc_dir, 'hoi_list.json')
    hoi_list = io.load_json_object(hoi_list_json)

    # Load subset ids to eval on
    split_ids_json = os.path.join(args.proc_dir, 'split_ids.json')
    split_ids = io.load_json_object(split_ids_json)
    global_ids = split_ids[args.subset]
    global_ids_set = set(global_ids)

    # Create gt_dets
    print('Creating GT dets ...')
    gt_dets = load_gt_dets(args.proc_dir, global_ids_set)

    print(f'Starting a pool of {args.num_processes} workers ...')
    p = Pool(args.num_processes)

    fp_inc = np.zeros(600, dtype=np.float32)
    fp_bck = np.zeros(600, dtype=np.float32)
    fp_Hmis = np.zeros(600, dtype=np.float32)
    fp_Omis = np.zeros(600, dtype=np.float32)
    fp_dupl = np.zeros(600, dtype=np.float32)
    fp_misg = np.zeros(600, dtype=np.float32)
    fp_occl = np.zeros(600, dtype=np.float32)
    tp_ = np.zeros(600, dtype=np.float32)
    rec = np.zeros(600, dtype=np.float32)
    prec = np.zeros(600, dtype=np.float32)
    ndet = np.zeros(600, dtype=np.float32)
    npos = np.zeros(600, dtype=np.float32)

    eval_inputs = []
    for hoi in hoi_list:
        eval_inputs.append(
            (hoi['id'], global_ids, gt_dets, args.pred_hoi_dets_hdf5, args.out_dir))

    print(f'Begin mAP computation ...')
    output = p.starmap(eval_hoi, eval_inputs)
    # output = eval_hoi('003',global_ids,gt_dets,args.pred_hoi_dets_hdf5,args.out_dir)

    p.close()
    p.join()

    mAP = {
        'AP': {},
        'mAP': 0,
        'invalid': 0,
    }
    map_ = 0
    count = 0
    for ap, hoi_id, errors in output:
        mAP['AP'][hoi_id] = ap
        if not np.isnan(ap):
            count += 1
            map_ += ap

        hoi_idx = int(hoi_id) - 1
        fp_inc[hoi_idx] = errors['fp_inc']
        fp_bck[hoi_idx] = errors['fp_bck']
        fp_Hmis[hoi_idx] = errors['fp_Hmis']
        fp_Omis[hoi_idx] = errors['fp_Omis']
        fp_dupl[hoi_idx] = errors['fp_dupl']
        fp_misg[hoi_idx] = errors['fp_misg']
        fp_occl[hoi_idx] = errors['fp_occl']
        tp_[hoi_idx] = errors['tp']
        rec[hoi_idx] = errors['rec']
        prec[hoi_idx] = errors['prec']
        ndet[hoi_idx] = errors['ndet']
        npos[hoi_idx] = errors['npos']

    mAP['mAP'] = map_ / count
    mAP['invalid'] = len(output) - count

    mAP_json = os.path.join(
        args.out_dir,
        'mAP.json')
    io.dump_json_object(mAP, mAP_json)

    print(f'APs have been saved to {args.out_dir}')
    print(
        '--------------------------------------------Reporting Error Analysis (%)-----------------------------------------------')
    print('{: >27} {:} {:} {:} {:} {:} {:}'.format(' ', 'inc', 'bck', 'H_mis', 'O_mis', 'mis-gr', 'occl'))

    for i in range(600):
        print(
            '{: >23}: {:6.2f} {:4.2f} {:4.2f} {:5.2f} {:5.2f} {:5.2f} (rec:{:5.2f} = #tp:{:4d}/#pos:{:4d}) (prec:{:5.2f} = #tp:{:4d}/#det:{:4d})'.format(
                hoi_list[i]['verb'] + '_' + hoi_list[i]['object'],
                fp_inc[i] * 100.0,
                fp_bck[i] * 100.0,
                fp_Hmis[i] * 100.0,
                fp_Omis[i] * 100.0,
                fp_misg[i] * 100.0,
                fp_occl[i] * 100.0,
                rec[i] * 100.0,
                int(tp_[i]),
                int(npos[i]),
                prec[i] * 100.0,
                int(tp_[i]),
                int(ndet[i])))


if __name__ == '__main__':
    main()
