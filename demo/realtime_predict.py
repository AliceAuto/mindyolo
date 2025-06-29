import argparse
import cv2
import os
import sys
import time
import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore import nn

from demo.predict import detect, segment, set_default_infer, get_parser_infer, parse_args
from mindyolo.models import create_model
from mindyolo.utils import logger, set_seed
from mindyolo.utils.utils import draw_result

def get_parser_realtime(parents=None):
    base_parser = get_parser_infer()
    parser = argparse.ArgumentParser(
        description='实时摄像头推理参数',
        parents=[base_parser],
        add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--device_id', type=int, default=0, help='推理设备ID')
    parser.set_defaults(image_path=None, save_result=False)
    parser.add_argument('--camera_index', type=int, default=0, help='摄像头设备索引')
    parser.add_argument('--show_fps', type=bool, default=True, help='是否显示帧率')
    parser.add_argument('--frame_size', type=int, nargs=2, default=[640, 480], 
                       help='摄像头分辨率 [宽, 高]')
    parser.add_argument('--exit_key', type=str, default='q', help='退出程序的按键')
    return parser

def infer(args, network, frame=None):
    # 直接使用传入frame作为输入图像，无需重新读取或创建模型
    if frame is not None:
        img = frame
    elif isinstance(args.image_path, np.ndarray):
        img = args.image_path
    else:
        if not os.path.exists(args.image_path):
            raise ValueError(f"输入文件不存在: {args.image_path}")
        img = cv2.imread(args.image_path)

    is_coco_dataset = "coco" in args.data.dataset_name if hasattr(args.data, 'dataset_name') else False

    if args.task == "detect":
        result_dict = detect(
            network=network,
            img=img,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            conf_free=args.conf_free,
            exec_nms=args.exec_nms,
            nms_time_limit=args.nms_time_limit,
            img_size=args.img_size,
            stride=max(max(args.network.stride), 32),
            num_class=args.data.nc,
            is_coco_dataset=is_coco_dataset,
        )
        plot_img = draw_result(img, result_dict, args.data.names, is_coco_dataset=is_coco_dataset)

        if args.save_result:
            save_dir = os.path.join(args.save_dir, "detect_results")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"result_{int(time.time())}.jpg")
            if plot_img is not None:
                cv2.imwrite(save_path, plot_img)
            else:
                logger.error("保存失败，输出图像为空")

    elif args.task == "segment":
        result_dict = segment(
            network=network,
            img=img,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            conf_free=args.conf_free,
            nms_time_limit=args.nms_time_limit,
            img_size=args.img_size,
            stride=max(max(args.network.stride), 32),
            num_class=args.data.nc,
            is_coco_dataset=is_coco_dataset,
        )
        plot_img = draw_result(img, result_dict, args.data.names, is_coco_dataset=is_coco_dataset)

        if args.save_result:
            save_dir = os.path.join(args.save_dir, "segment_results")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"result_{int(time.time())}.jpg")
            cv2.imwrite(save_path, plot_img)
            logger.info(f'分割结果已保存至：{save_path}')

    else:
        raise ValueError(f"不支持的任务类型: {args.task}")

    return plot_img


def realtime_infer(args):
    set_seed(args.seed)
    set_default_infer(args)

    # 创建模型，只做一次
    network = create_model(
        model_name=args.network.model_name,
        model_cfg=args.network,
        num_classes=args.data.nc,
        sync_bn=False,
        checkpoint_path=args.weight,
    )
    network.set_train(False)
    ms.amp.auto_mixed_precision(network, amp_level=args.ms_amp_level)

    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.frame_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.frame_size[1])

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("无法读取摄像头帧，退出循环")
            break

        start_time = time.time()
        processed_frame = infer(args, network, frame=frame)
        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0

        if args.show_fps:
            cv2.putText(processed_frame, f'FPS: {fps:.2f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Live Detection', processed_frame)

        if cv2.waitKey(1) == ord(args.exit_key):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("实时推理结束")


if __name__ == "__main__":
    parser = get_parser_realtime()
    args = parse_args(parser)
    realtime_infer(args)
