"""
Date: 12/10/2023
@author: bmhungqb
"""
import argparse
import pickle
import os
from pathlib import Path
import pandas as pd
from collections import Counter
import warnings
import torch
import torch.backends.cudnn as cudnn

warnings.filterwarnings('ignore')
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_boxes, check_requirements, cv2,
                                      check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args,
                                      check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.models.common import DetectMultiBackend
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT
from models.glpdepth import GLP
from models.lstm import LSTM
from models.extract_features_depth_map import ExtractFeature
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

ROOT = '.'
WEIGHTS = ROOT + '/weights'

@torch.no_grad()
def run(
        glp_depth_weights = 'vinvino02/glpn-kitti',
        type_distance_predictor = 'lstm',
        distance_estimate_weights= WEIGHTS + 'lstm_model.pth',
        scaler_distance_estimate= WEIGHTS + '/lstm_scaler.pkl',
        source='0',
        yolo_weights= WEIGHTS + '/yolov5x.pt',  # models.pt path(s),
        strong_sort_weights= WEIGHTS + '/osnet_x0_25_msmt17.pt',  # models.pt path,
        config_strongsort= ROOT + '/strong_sort/configs/strong_sort.yaml',
        imgsz=(375, 1242),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT + '/runs/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        hide_distance=False, # hidde distance
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        count=False,  # get counts of every objects
        draw=False,  # draw object trajectory lines

):
    source = str(source)
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo models
        exp_name = yolo_weights
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + strong_sort_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    device = select_device(device)

    # LOAD MODEL YOLOv5
    model_detect = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, names, pt = model_detect.stride, model_detect.names, model_detect.pt
    print('*DONE-Load models yolov5*')

    # LOAD MODEL GLPdepth
    model_depth = GLP(glp_depth_weights)
    print('*DONE-Load models GLPdepth*')

    # LOAD MODEL DISTANCE PREDICTOR
    if type_distance_predictor == 'lstm':
        model_de = LSTM(distance_estimate_weights)
        de_scaler = pickle.load(open(scaler_distance_estimate, 'rb'))
    else:
        model_de = LSTM(distance_estimate_weights)
        de_scaler = pickle.load(open(scaler_distance_estimate, 'rb'))
    print('*DONE-Load distance predictor*')

    # LOAD  EXTRACT FEATURE
    extractor = ExtractFeature()
    print('*DONE-Load extract feature depth map*')

    trajectory = {}
    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # INIT STRONGSORT
    cfg = get_config()
    cfg.merge_from_file(opt.config_strongsort)

    # Create as many strong sort instances as there are video sources
    strongsort_list = []
    for i in range(nr_sources):
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,
            )
        )
    outputs = [None] * nr_sources

    # Run tracking
    model_detect.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        pred = model_detect(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            if cfg.STRONGSORT.ECC:  # camera motion compensation
                strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                # distance estimate
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                xyxys = det[:, 0:4]
                confs = det[:, 4]
                clss = det[:, 5]
                # pass detections to strongsort
                t4 = time_sync()
                outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4
                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    # pass detection to distance predictor
                    depth_map = model_depth.predict(im0s, im0s.shape)
                    data_feature_de = extractor.make_dataset(xyxys, depth_map)
                    data_feature_de_scaler = torch.tensor(de_scaler.transform(data_feature_de), dtype=torch.float32)
                    distances = model_de.predict(data_feature_de_scaler).flatten()
                    for j, (output, conf) in enumerate(zip(outputs[i], confs)):
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        distance = distances[j]
                        if draw:
                            # object trajectory
                            center = ((int(bboxes[0]) + int(bboxes[2])) // 2, (int(bboxes[1]) + int(bboxes[3])) // 2)
                            if id not in trajectory:
                                trajectory[id] = []
                            trajectory[id].append(center)
                            for i1 in range(1, len(trajectory[id])):
                                if trajectory[id][i1 - 1] is None or trajectory[id][i1] is None:
                                    continue
                                # thickness = int(np.sqrt(1000/float(i1+10))*0.3)
                                thickness = 2
                                try:
                                    cv2.line(im0, trajectory[id][i1 - 1], trajectory[id][i1], (0, 0, 255), thickness)
                                except:
                                    pass

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 11 + '\n') % (frame_idx + 1, cls, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))

                        if save_vid or save_crop or show_vid:  # Add bbox to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                                                  (
                                                                      f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                            label = None if hide_labels else (
                                f'{id} {names[c]} {distance: .2f}' if not hide_distance and hide_conf else (
                                    f'{id} {conf:.2f} {distance: .2f}' if not hide_distance and hide_class else (
                                        f'{id} {names[c]} {conf:.2f} {distance: .2f}' if not hide_distance else (
                                            f'{id} {names[c]}' if hide_conf else (
                                                f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'
                                            )
                                        )
                                    )
                                )
                            )

                            annotator.box_label(bboxes, label, color=colors(c, True))

                            if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[
                                    c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')

            else:
                strongsort_list[i].increment_ages()
                LOGGER.info('No detections')

            if count:
                itemDict = {}
                ## NOTE: this works only if save-txt is true
                try:
                    df = pd.read_csv(txt_path + '.txt', header=None, delim_whitespace=True)
                    df = df.iloc[:, 0:3]
                    df.columns = ["frameid", "class", "trackid"]
                    df = df[['class', 'trackid']]
                    df = (df.groupby('trackid')['class']
                          .apply(list)
                          .apply(lambda x: sorted(x))
                          ).reset_index()

                    df.colums = ["trackid", "class"]
                    df['class'] = df['class'].apply(lambda x: Counter(x).most_common(1)[0][0])
                    vc = df['class'].value_counts()
                    vc = dict(vc)

                    vc2 = {}
                    for key, val in enumerate(names):
                        vc2[key] = val
                    itemDict = dict((vc2[key], value) for (key, value) in vc.items())
                    itemDict = dict(sorted(itemDict.items(), key=lambda item: item[0]))
                    # print(itemDict)

                except:
                    pass

                if save_txt:
                    ## overlay
                    display = im0.copy()
                    h, w = im0.shape[0], im0.shape[1]
                    x1 = 10
                    y1 = 10
                    x2 = 10
                    y2 = 70

                    txt_size = cv2.getTextSize(str(itemDict), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                    cv2.rectangle(im0, (x1, y1 + 1), (txt_size[0] * 2, y2), (0, 0, 0), -1)
                    cv2.putText(im0, '{}'.format(itemDict), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (210, 210, 210), 2)
                    cv2.addWeighted(im0, 0.7, display, 1 - 0.7, 0, im0)

            # # current frame // tesing
            # cv2.imwrite('testing.jpg', im0)

            if show_vid:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    break

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(
        f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update models (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--glp-depth-weights', type=str, default='vinvino02/glpn-kitti', help='models.bin path')
    parser.add_argument('--type-distance-predictor', type=str, default='lstm', help='lstm')
    parser.add_argument('--distance-estimate-weights', type=str, default=WEIGHTS + '/lstm_model.pth', help='models.pth path(s)')
    parser.add_argument('--scaler-distance-estimate', type=str, default=WEIGHTS + '/lstm_scaler.pkl', help='scaler path')
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS + '/yolov5x.pt', help='models.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS + '/osnet_x0_25_msmt17.pth')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--count', action='store_true', help='display all MOT counts results on screen')
    parser.add_argument('--draw', action='store_true', help='display object trajectory lines')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT + '/runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--hide-distance', default=False, action='store_true', help='hide distance')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT + '/requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)