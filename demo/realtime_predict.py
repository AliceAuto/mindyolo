import argparse
import cv2
import os
import sys
from pathlib import Path
import time
from predict import infer
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
    # 添加设备ID参数
    parser.add_argument('--device_id', type=int, default=0, help='推理设备ID')
    parser.set_defaults(image_path=None, save_result=False)
    parser.add_argument('--camera_index', type=int, default=0, help='摄像头设备索引')
    parser.add_argument('--show_fps', type=bool, default=True, help='是否显示帧率')
    parser.add_argument('--frame_size', type=int, nargs=2, default=[640, 480], 
                       help='摄像头分辨率 [宽, 高]')
    parser.add_argument('--exit_key', type=str, default='q', help='退出程序的按键')
    return parser

# 在实时推理循环中禁用保存功能
def realtime_infer(args):
    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.frame_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.frame_size[1])
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 添加FPS计算
        start_time = time.time()
        
        # 执行推理并绘制结果
        processed_frame = infer(args, frame=frame)
        
        # 添加FPS显示（关键修改）
        if args.show_fps:
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(processed_frame, f'FPS: {fps:.2f}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        cv2.imshow('Live Detection', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord(args.exit_key):
            break

    # 资源释放（保持不变）
    cap.release()
    cv2.destroyAllWindows()
    logger.info("实时推理结束")

if __name__ == "__main__":
    # 合并基础参数和实时参数
    base_parser = get_parser_realtime()
    args = parse_args(base_parser)
    realtime_infer(args)