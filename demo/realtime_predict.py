import argparse
import cv2
import os
import sys
from pathlib import Path
import time

import mindspore as ms
from demo.predict import detect, segment, set_default_infer, get_parser_infer
from mindyolo.models import create_model
from mindyolo.utils import logger, set_seed
from mindyolo.utils.config import parse_args

def get_parser_realtime(parents=None):
    # 继承基础参数解析器并扩展
    base_parser = get_parser_infer()
    parser = argparse.ArgumentParser(
        description='实时摄像头推理参数',
        parents=[base_parser],
        add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.set_defaults(image_path=None, save_result=False)
    parser.add_argument('--camera_index', type=int, default=0, help='摄像头设备索引')
    parser.add_argument('--show_fps', type=bool, default=True, help='是否显示帧率')
    parser.add_argument('--frame_size', type=int, nargs=2, default=[640, 480], 
                       help='摄像头分辨率 [宽, 高]')
    parser.add_argument('--exit_key', type=str, default='q', help='退出程序的按键')
    return parser

def realtime_infer(args):
    # 初始化随机种子和推理配置
    set_seed(args.seed)
    set_default_infer(args)

    # 创建网络模型
    network = create_model(
        model_name=args.network.model_name,
        model_cfg=args.network,
        num_classes=args.data.nc,
        sync_bn=False,
        checkpoint_path=args.weight,
    )
    network.set_train(False)
    ms.amp.auto_mixed_precision(network, amp_level=args.ms_amp_level)

    # 初始化视频捕获
    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.frame_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.frame_size[1])

    # 帧率计算相关
    fps = 0
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("摄像头读取失败")
            break

        # 执行推理
        if args.task == "detect":
            result = detect(
                network=network,
                img=frame,
                conf_thres=args.conf_thres,
                iou_thres=args.iou_thres,
                img_size=args.img_size,
                stride=max(max(args.network.stride), 32),
                num_class=args.data.nc,
            )
        elif args.task == "segment":
            result = segment(
                network=network,
                img=frame,
                conf_thres=args.conf_thres,
                iou_thres=args.iou_thres,
                img_size=args.img_size,
                stride=max(max(args.network.stride), 32),
                num_class=args.data.nc,
            )

        # 在画面上绘制结果
        viz_frame = frame.copy()
        for box, score, cls_id in zip(result['bbox'], result['score'], result['category_id']):
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(viz_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(viz_frame, f'{args.data.names[cls_id]} {score:.2f}', 
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 显示帧率
        if args.show_fps:
            frame_count += 1
            if frame_count >= 10:
                fps = frame_count / (time.time() - start_time)
                start_time = time.time()
                frame_count = 0
            cv2.putText(viz_frame, f'FPS: {fps:.1f}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 显示画面
        cv2.imshow('Realtime Detection', viz_frame)

        # 退出机制
        if cv2.waitKey(1) & 0xFF == ord(args.exit_key):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    logger.info("实时推理结束")

if __name__ == "__main__":
    # 合并基础参数和实时参数
    base_parser = get_parser_realtime()
    args = parse_args(base_parser)
    realtime_infer(args)