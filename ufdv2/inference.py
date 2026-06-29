import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import subprocess
# =====================================================
# ⚠️ 路径保留你本机原有配置
ROOT_DIR = os.path.dirname(
    os.path.dirname(__file__)
)
FFMPEG_PATH = os.path.join(
    ROOT_DIR,
    "ffmpeg-8.1.1-essentials_build",
    "bin",
    "ffmpeg.exe"
)
WEIGHT_PATH = os.path.join(
    ROOT_DIR,
    "culane_res18.pth"
)
# =====================================================
# 全局空变量，延后初始化
DEVICE = None
CFG = None
MODEL = None
IMG_TRANSFORMS = None

def get_ufld_cfg():
    cfg_path = os.path.join(
        os.path.dirname(__file__),
        "configs",
        "culane_res18.py"
    )
    from ufdv2.utils.config import Config
    cfg = Config.fromfile(cfg_path)
    cfg.batch_size = 1
    cfg.use_aux = False
    cfg.tta = False
    cfg.test_model = WEIGHT_PATH
    return cfg

# 新增：懒加载初始化模型（只有第一次推理才加载权重）
def init_model_once():
    global DEVICE,CFG,MODEL,IMG_TRANSFORMS
    if MODEL is not None:
        return
    print("第一次调用，开始加载模型权重")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CFG = get_ufld_cfg()
    from ufdv2.utils.common import get_model
    MODEL = get_model(CFG)
    CFG.row_anchor = np.linspace(0, 1, CFG.num_row)
    checkpoint = torch.load(CFG.test_model, map_location=DEVICE)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    MODEL.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()}, strict=False)
    MODEL.to(DEVICE)
    MODEL.eval()
    IMG_TRANSFORMS = transforms.Compose([
        transforms.Resize((CFG.train_height, CFG.train_width)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:3] if x.shape[0] > 3 else x),  # 强制取前3通道
        transforms.Normalize((0.485, 0.406, 0.456), (0.229, 0.224, 0.225))  # 注意：标准RGB均值
        # transforms.Normalize((0.485, 0.406),(0.229,0.224,0.225))
    ])

def pred2coords(pred, cfg, img_w, img_h, local_width=1):
    coords = []
    loc_row = pred["loc_row"].softmax(1).cpu()
    valid_row = pred["exist_row"].argmax(1).cpu()
    num_grid = loc_row.shape[1]
    for lane_idx in range(cfg.num_lanes):
        lane = []
        if valid_row[0, :, lane_idx].sum() < cfg.num_row * 0.6:
            continue
        for r in range(cfg.num_row):
            if not valid_row[0, r, lane_idx]:
                continue
            idx = loc_row[0, :, r, lane_idx].argmax().item()
            left = max(0, idx - local_width)
            right = min(num_grid - 1, idx + local_width)
            inds = torch.arange(left, right + 1)
            probs = loc_row[0, inds, r, lane_idx]
            exp_x = (probs * inds.float()).sum() / probs.sum()
            x = int(exp_x / (num_grid - 1) * img_w)
            y = int(cfg.row_anchor[r] * img_h)
            lane.append((x, y))
        if len(lane) > 6:
            coords.append(lane)
    return coords

# 单图推理
# def run_ufld_on_image(img_input):
#     init_model_once() # 关键：进入函数才加载模型
#     print("=====进入模型推理函数=====")
#     if isinstance(img_input, str):
#         img_bgr = cv2.imread(img_input)
#         if img_bgr is None:
#             raise ValueError("Image read failed")
#     else:
#         img_bgr = img_input
#     H, W = img_bgr.shape[:2]
#     crop_y = int(H * (1 - CFG.crop_ratio))
#     img_crop = img_bgr[crop_y:, :]
#     img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
#     img_pil = Image.fromarray(img_rgb)
#     img_tensor = IMG_TRANSFORMS(img_pil).unsqueeze(0).to(DEVICE)
#     with torch.no_grad():
#         pred = MODEL(img_tensor)
#     lanes = pred2coords(pred, CFG, W, H - crop_y)
#     vis = img_bgr.copy()
#     for lane in lanes:
#         pts = np.array([(x,y) for x,y in lane], dtype=np.int32)
#         cv2.polylines(vis,[pts],False,(0,255,0),3)
#     return vis

def run_ufld_on_image(img_input):
    init_model_once()
    print("=====进入模型推理函数=====")

    if isinstance(img_input, str):
        img_bgr = cv2.imread(img_input)
        if img_bgr is None:
            raise ValueError("Image read failed")
    else:
        img_bgr = img_input

    H, W = img_bgr.shape[:2]
    crop_y = int(H * (1 - CFG.crop_ratio)) if hasattr(CFG, 'crop_ratio') else 0
    img_crop = img_bgr[crop_y:, :]

    # 关键修复：确保是RGB 3通道
    img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb).convert('RGB')  # 强制RGB

    img_tensor = IMG_TRANSFORMS(img_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = MODEL(img_tensor)

    lanes = pred2coords(pred, CFG, W, H - crop_y)

    vis = img_bgr.copy()
    for lane in lanes:
        pts = np.array([(x, y + crop_y) for x, y in lane], dtype=np.int32)  # 注意坐标偏移
        cv2.polylines(vis, [pts], False, (0, 255, 0), 5)

    return vis

# 视频推理
def run_ufld_on_video(video_path, save_path):
    init_model_once()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <=0 or fps>120:
        fps=25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    avi_path = os.path.splitext(save_path)[0]+".avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(avi_path,fourcc,fps,(W,H))
    while True:
        ret,frame = cap.read()
        if not ret:break
        vis = run_ufld_on_image(frame)
        out.write(vis)
    cap.release()
    out.release()
    cmd = [FFMPEG_PATH,"-y","-i",avi_path,"-vcodec","libx264","-pix_fmt","yuv420p",save_path]
    subprocess.run(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if os.path.exists(avi_path):
        os.remove(avi_path)
