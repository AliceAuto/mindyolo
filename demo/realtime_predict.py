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
    parser.add_argument('--show_fps', type=bool, default=True, help='是否显示帧率')
    parser.add_argument('--frame_size', type=int, nargs=2, default=[640, 480], 
                       help='摄像头分辨率 [宽, 高]')
    parser.add_argument('--exit_key', type=str, default='q', help='退出程序的按键')
    parser.add_argument('--mode', type=str, default='live', choices=['video', 'image', 'live'], help='推理模式: 视频(video)、图片(image)或直播(live)。视频模式需指定--video_path，图片模式需指定--image_path')
    parser.add_argument('--video_path', type=str, default=None, help='Path to input video file')
    parser.add_argument('--output_video', type=str, default=None, help='Path to output video file')
    parser.add_argument('--save_frames', type=bool, default=False, help='是否保存生成的帧图片（以result_xxx命名）')
    parser.add_argument('--pad_if_mismatch', type=bool, default=False, help='当尺寸不匹配时，是否对原图像进行填充推理后再剪裁复原')
    parser.add_argument('--camera_index', type=str, default='0', help='摄像头索引或视频流URL')
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
    
    # 验证输入尺寸与模型配置一致
    model_img_size = args.network.img_size if hasattr(args.network, 'img_size') else args.img_size
    if args.frame_size[0] != model_img_size or args.frame_size[1] != model_img_size:
        logger.warning(f"推理资源尺寸不匹配，模型期望 {model_img_size}x{model_img_size}，实际设置 {args.frame_size[0]}x{args.frame_size[1]}。将自动调整帧大小。")

    # 根据模式确定输入源
    if args.mode == "video":
        if not os.path.isfile(args.video_path):
            raise FileNotFoundError(f"视频文件不存在: {args.video_path}。请检查路径是否正确。")
        cap = cv2.VideoCapture(args.video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {args.video_path}。可能是文件格式不受支持或文件已损坏。")
        # 创建保存帧图片的目录
        if args.save_frames:
            os.makedirs('detect_results', exist_ok=True)
        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"正在处理视频文件: {args.video_path}, 分辨率: {frame_width}x{frame_height}, FPS: {fps:.2f}")
    elif args.mode == 'live':
        cap = cv2.VideoCapture(args.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.frame_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.frame_size[1])
        fps = 30  # 摄像头默认FPS
        frame_width, frame_height = args.frame_size
        logger.info(f"正在使用摄像头 {args.camera_index}, 分辨率: {frame_width}x{frame_height}")
    elif args.mode == 'image':
        if not args.image_path:
            raise ValueError("使用图片模式时必须指定 --image_path 参数")
        if not os.path.exists(args.image_path):
            raise ValueError(f"图片文件不存在: {args.image_path}")
        frame = cv2.imread(args.image_path)
        frame = cv2.resize(frame, tuple(args.frame_size))
        processed_frame = infer(args, network, frame=frame)
        if args.output_video:
            output_dir = os.path.dirname(args.output_video)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(args.output_video, processed_frame)
            logger.info(f"图片结果已保存至: {args.output_video}")
        else:
            cv2.imshow('Detection Result', processed_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return
    else:
        raise ValueError(f"不支持的推理模式: {args.mode}")

    # 初始化视频写入器（如果需要）
    video_writer = None
    if args.output_video:
        # 创建输出目录
        output_dir = os.path.dirname(args.output_video)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        # 获取输出视频编码
        # 尝试多种编码以提高兼容性
    # 根据文件扩展名选择兼容编码
    ext = os.path.splitext(args.output_video)[1].lower()
    if ext == '.mp4':
        codecs = ['mp4v', 'avc1', 'H264', 'MPEG']  # 优先尝试更兼容的MP4编码
    elif ext == '.avi':
        codecs = ['XVID']  # AVI兼容编码
    else:
        codecs = ['avc1', 'mp4v', 'XVID']  # 默认尝试所有编码
    
    # 尝试多种编码以提高兼容性
    video_writer = None
    for codec in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            video_writer = cv2.VideoWriter(args.output_video, fourcc, fps, (args.frame_size[0], args.frame_size[1]))
            if video_writer.isOpened():
                logger.info(f"使用{codec}编码创建视频写入器成功")
                break
        except Exception as e:
            logger.warning(f"使用{codec}编码失败: {e}")
    
    if not video_writer or not video_writer.isOpened():
        raise RuntimeError(f"所有编码尝试({', '.join(codecs)})均失败，无法创建视频写入器: {args.output_video}")
    output_abs_path = os.path.abspath(args.output_video)
    logger.info(f"视频结果将保存至: {output_abs_path}")

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.video_path:
                    logger.info(f"视频处理完成，共处理 {frame_count} 帧")
                else:
                    logger.warning("无法读取摄像头帧，退出循环")
                break

            # 调整帧大小以匹配模型输入要求
            if args.pad_if_mismatch:
                # 计算填充比例
                h, w = frame.shape[:2]
                scale = min(model_img_size / w, model_img_size / h)
                new_w, new_h = int(w * scale), int(h * scale)
                # 调整图像大小
                frame_resized = cv2.resize(frame, (new_w, new_h))
                # 计算填充量
                delta_w = model_img_size - new_w
                delta_h = model_img_size - new_h
                top, bottom = delta_h // 2, delta_h - (delta_h // 2)
                left, right = delta_w // 2, delta_w - (delta_w // 2)
                # 添加填充
                frame = cv2.copyMakeBorder(frame_resized, top=top, bottom=bottom, left=left, right=right, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
                # 保存填充信息用于后续剪裁
                pad_info = (top, bottom, left, right, h, w)
            else:
                frame = cv2.resize(frame, (model_img_size, model_img_size))
                pad_info = None

            start_time = time.time()
            processed_frame = infer(args, network, frame=frame)
            # 如果使用了填充，剪裁回原始尺寸
            if args.pad_if_mismatch and pad_info is not None:
                top, bottom, left, right, h, w = pad_info
                # 裁剪填充区域并恢复原始尺寸
                processed_frame = processed_frame[top:top+new_h, left:left+new_w]
                processed_frame = cv2.resize(processed_frame, (w, h))
            elapsed = time.time() - start_time
            display_fps = 1.0 / elapsed if elapsed > 0 else 0
            frame_count += 1

            # 显示FPS
            if args.show_fps:
                cv2.putText(processed_frame, f'FPS: {display_fps:.2f}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 显示处理结果
            cv2.imshow('Detection Result', processed_frame)

            # 写入视频文件
            if video_writer:
                if processed_frame is None:
                    logger.error(f"第 {frame_count} 帧处理结果为空，跳过写入")
                    continue
                # 检查帧格式和通道数
                if len(processed_frame.shape) != 3 or processed_frame.shape[2] not in [3, 4]:
                    logger.error(f"第 {frame_count} 帧格式无效，通道数: {processed_frame.shape[2] if len(processed_frame.shape) == 3 else '未知'}")
                    continue
                # 转换RGBA为BGR格式
                if processed_frame.shape[2] == 4:
                    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGBA2BGR)
                # 检查帧大小是否匹配视频写入器设置
                if processed_frame.shape[1] != args.frame_size[0] or processed_frame.shape[0] != args.frame_size[1]:
                    logger.warning(f"帧大小不匹配，调整为视频写入器大小: {args.frame_size}")
                    processed_frame = cv2.resize(processed_frame, tuple(args.frame_size))
                success = video_writer.write(processed_frame)
                if not success:
                    logger.error(f"写入第 {frame_count} 帧到视频文件失败")
                if frame_count % 100 == 0:
                    logger.info(f"已处理 {frame_count} 帧视频")

            # 退出条件：按指定键或视频结束
            key = cv2.waitKey(1) if args.video_path else cv2.waitKey(1)
            if key == ord(args.exit_key) or (args.video_path and not ret):
                # 保存推理结果图片
                if args.save_frames:
                    save_path = os.path.join('detect_results', f'result_{frame_count}.jpg')
                    cv2.imwrite(save_path, frame)
                    logger.info(f'已保存推理结果图片: {save_path}')

                # 退出条件：按指定键或视频结束
                key = cv2.waitKey(1) if args.video_path else cv2.waitKey(1)
                if key == ord(args.exit_key) or (args.video_path and not ret):
                    break

    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}", exc_info=True)
    finally:
        # 释放资源
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        logger.info("推理过程已结束")


if __name__ == "__main__":
    parser = get_parser_realtime()
    args = parse_args(parser)
    realtime_infer(args)
