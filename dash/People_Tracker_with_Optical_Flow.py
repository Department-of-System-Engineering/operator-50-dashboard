import cv2
import os
import time
import threading
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque, defaultdict
from ultralytics import YOLO
import matplotlib.pyplot as plt
import ast

# =================== Feature falg ===================
SAVE_DATA = True
save_state = "Deactivate saving data"
last_saved = "No data saved"
DASHBOARD_ACTIVE = True
dashboard_state = "Deactivate dashboard"

# =================== Parameters ===================
df = pd.read_excel("param.xlsx")
params = {row["param"]: row["value"] for _, row in df.iterrows()}
for key, val in params.items():
    if isinstance(val, str):
        try:
            params[key] = ast.literal_eval(val)
        except:
            pass

WIN = params["win_size"]
LEVELS = params["pyramid_levels"]
GRID_SPACING = params["grid_spacing"]
NEIGH_RADIUS = params["neighbor_radius"]
FB_MAX_ERR = params["fb_max_error"]
PREDICT_WINDOW = params["predict_window"]
PREDICT_HORIZON = params["predict_horizon"]
PREDICT_MIN_POINTS = params["predict_min_points"]
TEXT_COLOR = params["text_color"]
POINT_DROP_RATIO = params["point_drop_ratio"]
TRAIL_LEN = params["trail_length"]
MEAN_DRIFT_THRESH = params["mean_drift_threshold"]
MAX_DISP = params["max_displacement"]
ASSIGN_DIST = params["assign_distance"]
REINIT_WINDOW = params["reinit_window"]
VISFRAME = params["vis_frame_count"]
N_SAVE_FRAME = params["n_save_frame"]

REINIT_ACTIVE = {}
NEXT_ID = 0
id_frame_count = 0
current_filename = None
ID_COLORS = {}
track_buffer = []

lk_params = dict(winSize=(WIN, WIN), maxLevel=LEVELS,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1, 1e-10))

# =================== Dashboard data ===================
dashboard_data = {
    "frames": deque(maxlen=1000),
    "positions": defaultdict(lambda: {"x": deque(maxlen=VISFRAME), "y": deque(maxlen=VISFRAME)}),
    "points": defaultdict(lambda: deque(maxlen=VISFRAME)),
    "reinits": defaultdict(list)
}


def show_fullscreen(window_name, image, screen_w=1920, screen_h=1080):
    h, w = image.shape[:2]
    scale = min(screen_w / w, screen_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    y0 = (screen_h - new_h) // 2
    x0 = (screen_w - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, canvas)


def merge_frame_and_dashboard(frame, dashboard):
    h = max(frame.shape[0], dashboard.shape[0])
    w1 = frame.shape[1]
    w2 = dashboard.shape[1]
    if frame.shape[0] != h:
        scale = h / frame.shape[0]
        frame = cv2.resize(frame, (int(w1 * scale), h))
    if dashboard.shape[0] != h:
        scale = h / dashboard.shape[0]
        dashboard = cv2.resize(dashboard, (int(w2 * scale), h))
    merged = np.concatenate((frame, dashboard), axis=1)
    bar_height = 100
    H, W = merged.shape[:2]
    canvas = np.zeros((H + bar_height, W, 3), dtype=np.uint8)
    canvas[:bar_height, :] = (255, 0, 0)
    canvas[bar_height:bar_height + H, :] = merged
    return canvas


def draw_dashboard_canvas(width=500, height=500, history=100, trackers=None):
    ids = list(dashboard_data["positions"].keys())
    n_ids = len(ids)
    subplot_width = width
    canvas = np.zeros((height, width + subplot_width, 3), dtype=np.uint8)

    for id_ in ids:
        pos = dashboard_data["positions"][id_]
        color = get_id_color(id_)

        if len(pos["x"]) > 1:
            pts = np.array(list(zip(pos["x"], pos["y"])), dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(canvas[:, :width], [pts], False, color, 2)

        if len(pos["x"]) > 0:
            x = int(pos["x"][-1])
            y = int(pos["y"][-1])
            cv2.circle(canvas[:, :width], (x, y), 5, color, -1)
            cv2.putText(canvas[:, :width], f"ID {id_}", (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    subplot_height = height // max(1, n_ids)

    all_values = []
    for id_ in ids:
        all_values.extend(dashboard_data["points"][id_])

    global_max = 1
    if trackers is not None:
        global_max = max((len(tr["trails"]) for tr in trackers), default=1)

    for i, id_ in enumerate(ids):
        values = list(dashboard_data["points"][id_])[-history:]
        if len(values) < 2:
            continue
        color = get_id_color(id_)
        plot = np.zeros((subplot_height, subplot_width, 3), dtype=np.uint8)

        scaled_y = [int(subplot_height - 1 - (v/global_max)*(subplot_height-1)) for v in values]

        for j in range(1, len(scaled_y)):
            x1 = int((j-1)/history*subplot_width)
            x2 = int(j/history*subplot_width)
            y1, y2 = scaled_y[j-1], scaled_y[j]
            cv2.line(plot, (x1,y1), (x2,y2), color, 2)

        values = list(dashboard_data["points"][id_])[-history:]
        window_len = len(values)
        window_start = id_frame_count - window_len
        for re_frame in dashboard_data["reinits"][id_]:
            if window_start <= re_frame < id_frame_count:
                rel = re_frame - window_start
                x_pos = int((rel / history) * subplot_width)
                use_overlay=True
                if use_overlay:
                    overlay = plot.copy()
                    cv2.line(overlay, (x_pos, 0), (x_pos, subplot_height), color, 10)
                    cv2.addWeighted(overlay, 0.4, plot, 0.6, 0, plot)
                else:
                    cv2.line(plot, (x_pos, 0), (x_pos, subplot_height), color, 10)

        y0 = i*subplot_height
        canvas[y0:y0+subplot_height, width:width+subplot_width] = plot


        cv2.putText(canvas, f"ID {id_}", (width+5, y0+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        shift_y=-30
        cv2.putText(canvas, f"{global_max}", (width+shift_y, y0+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 2)
        cv2.putText(canvas, f"{global_max//2}", (width+shift_y, y0+subplot_height//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 2)
        cv2.putText(canvas, "0", (width+shift_y, y0+subplot_height-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 2)

        if id_ == max(ids):
            cv2.putText(canvas, f"-{VISFRAME}", (width+2, y0+subplot_height-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 2)
            cv2.putText(canvas, f"-{VISFRAME//2}", (width+subplot_width//2-15, y0+subplot_height-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 2)
            cv2.putText(canvas, "0", (width+subplot_width-20, y0+subplot_height-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 2)

    return canvas


def save_tracks_to_excel_async():
    t = threading.Thread(target=save_tracks_to_excel, daemon=True)
    t.start()


def get_filename():
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"track_{now_str}.xlsx"


def save_tracks_to_excel():
    global track_buffer, current_filename, last_saved
    if not track_buffer:
        return

    if current_filename is None:
        current_filename = get_filename()

    df_new = pd.DataFrame(track_buffer, columns=["frame", "datetime", "id", "x", "y"])

    if os.path.exists(current_filename):
        with pd.ExcelWriter(current_filename, mode="a", engine="openpyxl", if_sheet_exists="overlay") as writer:
            startrow = writer.sheets['Sheet1'].max_row
            df_new.to_excel(writer, index=False, header=False, startrow=startrow)
    else:
        df_new.to_excel(current_filename, index=False)
    last_saved=id_frame_count
    print(f"Track data saved to: {current_filename}")

    track_buffer.clear()


def get_id_color(id):
    if id not in ID_COLORS:
        hsv = np.uint8([[[ (id*37) % 180 , 255, 255 ]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0,0]
        ID_COLORS[id] = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    return ID_COLORS[id]


def gen_colors(n):
    return [tuple(map(int, cv2.cvtColor(np.uint8([[[i*180//n,255,255]]]),
                                        cv2.COLOR_HSV2BGR)[0,0])) for i in range(n)]


def clamp_point(x,y,w,h):
    return float(np.clip(x,0,w-1)), float(np.clip(y,0,h-1))


def make_support_grid(cx,cy,radius,spacing,w,h):
    pts=[]
    x0=int(cx)-radius; y0=int(cy)-radius
    for yy in range(y0,y0+2*radius+1,spacing):
        for xx in range(x0,x0+2*radius+1,spacing):
            dx=xx-cx; dy=yy-cy
            if dx*dx+dy*dy <= radius*radius and 0<=xx<w and 0<=yy<h:
                pts.append([xx,yy])
    if not pts: return None
    return np.array(pts,dtype=np.float32).reshape(-1,1,2)


def forward_backward_filter(prev_gray, gray, p0):
    if p0 is None or len(p0)==0:
        return None,None
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
    if p1 is None: return None,None
    p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **lk_params)
    if p0r is None: return None,None

    p0f = p0.reshape(-1,2)
    p1f = p1.reshape(-1,2)
    p0rf = p0r.reshape(-1,2)

    fb = np.linalg.norm(p0f - p0rf, axis=1)
    disp = np.linalg.norm(p1f - p0f, axis=1)

    good = (st1.reshape(-1)>0) & (st2.reshape(-1)>0) \
           & (fb < FB_MAX_ERR) \
           & (disp < MAX_DISP)

    if not np.any(good):
        return None,None
    return p1f[good].reshape(-1,1,2), p0f[good].reshape(-1,1,2)

def predict_linear(hist,horizon,w,h):
    N=len(hist)
    if N<2: return []
    t=np.arange(N,dtype=np.float32)
    xy=np.array(hist,dtype=np.float32)
    x=xy[:,0]; y=xy[:,1]
    def fit(z):
        t0=t-t.mean(); z0=z-z.mean()
        slope=np.sum(t0*z0)/(np.sum(t0*t0)+1e-6)
        inter=z.mean()-slope*t.mean()
        return slope,inter
    sx,ix=fit(x); sy,iy=fit(y)
    preds=[]
    for k in range(1,horizon+1):
        tt=N-1+k
        px=sx*tt+ix; py=sy*tt+iy
        px,py=clamp_point(px,py,w,h)
        preds.append((px,py))
    return preds

def save_conflict_frame(frame, video_name, frame_count, trackers, detections, used_detections):
    out = frame.copy()
    for j, (cx, cy, p0, box) in enumerate(detections):
        x1, y1, x2, y2 = box
        color = (0,255,0) if j in used_detections else (0,0,255)
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        cv2.putText(out, f"{'OK' if j in used_detections else 'DROP'}",
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    fname = f"screenshot_{video_name}_frame{frame_count}_exp{len(trackers)}_det{len(detections)}.jpg"
    cv2.imwrite(fname, out)
    print(f"Screenshot saved to: {fname}")

def assign_detections_to_trackers(detections, trackers, frame, video_name, frame_count, ASSIGN_DIST,  manual=False):
    global NEXT_ID
    new_trackers = []
    used_trackers = set()
    used_detections = set()
    MANUAL_ASSIGN_DIST=100
    max_dist=MANUAL_ASSIGN_DIST if manual else ASSIGN_DIST
    
    if len(trackers) == 0:
        for cx, cy, p0, box in detections:
            tr = {
                "id": NEXT_ID,
                "cx": cx, "cy": cy, "p0": p0,
                "hist": deque(maxlen=PREDICT_WINDOW),
                "init_pts": len(p0),
                "trails": [deque(maxlen=TRAIL_LEN) for _ in range(len(p0))]
            }
            tr["hist"].append((cx, cy))
            new_trackers.append(tr)
            NEXT_ID += 1
        return new_trackers

    for i, tr in enumerate(trackers):
        best_j = -1
        best_d = 1e9
        for j, (cx, cy, p0, box) in enumerate(detections):
            if j in used_detections: 
                continue
            d = np.hypot(tr["cx"] - cx, tr["cy"] - cy)
            if d < best_d and d < max_dist:
                best_d = d
                best_j = j
        if best_j >= 0:
            cx, cy, p0, box= detections[best_j]
            tr["cx"], tr["cy"], tr["p0"] = cx, cy, p0
            tr["init_pts"] = len(p0)
            tr["trails"] = [deque(maxlen=TRAIL_LEN) for _ in range(len(p0))]
            tr["hist"].append((cx, cy))
            new_trackers.append(tr)
            used_trackers.add(i)
            used_detections.add(best_j)
        else:
            new_trackers.append(tr)

    if manual:
        for j, (cx, cy, p0, box) in enumerate(detections):
            if j not in used_detections:
                tr = {
                    "id": NEXT_ID,
                    "cx": cx, "cy": cy, "p0": p0,
                    "hist": deque(maxlen=PREDICT_WINDOW),
                    "init_pts": len(p0),
                    "trails": [deque(maxlen=TRAIL_LEN) for _ in range(len(p0))]
                }
                tr["hist"].append((cx, cy))
                new_trackers.append(tr)
                print(f"[Frame {frame_count}] New ID {NEXT_ID} added with manual initialization.")

                NEXT_ID += 1
                
    if not manual:
        for j, (cx, cy, _, _) in enumerate(detections):
            if j not in used_detections:
                print(f"YOLO detection ({cx:.1f},{cy:.1f}) discarded: too close to another or exceeding expected number of persons.")
        if any(j not in used_detections for j in range(len(detections))):
            save_conflict_frame(frame, video_name, frame_count, trackers, detections, used_detections)
    return new_trackers



# =================== Main ===================
def main():
    source= params["source"]
    video_name = f"camera{source}" if isinstance(source, int) else source.split('/')[-1].split('.')[0]
    cap=cv2.VideoCapture(source)       
    ok,frame=cap.read()
    if not ok: 
        print("Unable to open video"); 
        return
    h,w=frame.shape[:2]

    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt")
    print("YOLO model loaded")
    
    trackers = []

    prev_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame_count,t0=0,time.perf_counter()

    
    global NEXT_ID,id_frame_count, track_buffer, current_filename,SAVE_DATA,DASHBOARD_ACTIVE,save_state,dashboard_state,last_saved

    
    while True:
        ok,frame=cap.read()
        if not ok: break
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for tr in trackers:
            track_buffer.append([id_frame_count, timestamp, tr["id"], tr["cx"], tr["cy"]])

        if SAVE_DATA and (id_frame_count % N_SAVE_FRAME == 0 and id_frame_count > 0):
            save_tracks_to_excel_async()

        if DASHBOARD_ACTIVE:
            dashboard_data["frames"].append(id_frame_count)
            for tr in trackers:
                id_ = tr["id"]
                dashboard_data["positions"][id_]["x"].append(tr["cx"])
                dashboard_data["positions"][id_]["y"].append(tr["cy"])
                dashboard_data["points"][id_].append(len(tr["p0"]))
                if id_ in REINIT_ACTIVE:
                    dashboard_data["reinits"][id_].append(id_frame_count)

        if len(trackers)==0:
            results = model(frame, classes=[0], iou=0.4)
            detections = []
            for r in results:
                for box in r.boxes:
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    cx,cy = (x1+x2)/2, (y1+y2)/2
                    p0 = make_support_grid(cx,cy,NEIGH_RADIUS,GRID_SPACING,w,h)
                    if p0 is None:
                        p0=np.array([[[cx,cy]]],dtype=np.float32)
                    box=(x1,y1,x2,y2)
                    detections.append((cx, cy, p0, (x1,y1,x2,y2)))
            trackers = assign_detections_to_trackers(detections, trackers, frame, video_name, frame_count,ASSIGN_DIST)
                                                                    
        for ti,tr in enumerate(trackers):
            cx,cy,p0=tr["cx"],tr["cy"],tr["p0"]
            p1g,p0g=forward_backward_filter(prev_gray,gray,p0)

            if p1g is not None and len(p1g) > 0:
                pts = p1g.reshape(-1, 2)
                dists = np.hypot(pts[:,0] - cx, pts[:,1] - cy)
                k = max(1, int(len(dists) * 0.1))
                worst_mean = np.mean(np.sort(dists)[-k:])

                if (len(p1g) < POINT_DROP_RATIO * tr["init_pts"]) or (worst_mean > MEAN_DRIFT_THRESH):
                    reason = None
                    if len(p1g) < POINT_DROP_RATIO * tr["init_pts"]:
                        reason = f"too few points: ({len(p1g)}/{tr['init_pts']})"
                    elif worst_mean > MEAN_DRIFT_THRESH:
                        reason = f"mean drift too large ={worst_mean:.1f}px)"

                    current_frame = frame_count
                    if tr["id"] not in REINIT_ACTIVE:
                        REINIT_ACTIVE[tr["id"]] = current_frame + REINIT_WINDOW
                        print(f"[Frame {frame_count}] Tracker {tr['id']} reinit START: {reason}")
                    elif current_frame <= REINIT_ACTIVE[tr["id"]]:
                        print(f"[Frame {frame_count}] Tracker {tr['id']} retrying reinit ({reason})")
                        results = model(frame, classes=[0])
                        detections = []
                        for r in results:
                            for box in r.boxes:
                                x1,y1,x2,y2 = map(int, box.xyxy[0])
                                cx,cy = (x1+x2)/2, (y1+y2)/2
                                p0 = make_support_grid(cx,cy,NEIGH_RADIUS,GRID_SPACING,w,h)
                                if p0 is None: 
                                    p0 = np.array([[[cx,cy]]], dtype=np.float32)
                                detections.append((cx, cy, p0, (x1,y1,x2,y2)))
                        trackers = assign_detections_to_trackers(detections, trackers, frame, video_name, frame_count,ASSIGN_DIST)

                    else:
                        print(f"[Frame {frame_count}] Tracker {tr['id']} reinit FAILED (timeout)")
                        del REINIT_ACTIVE[tr["id"]]
                else:
                    disp=(p1g-p0g).reshape(-1,2)
                    dx,dy=np.mean(disp,axis=0)
                    cx,cy=clamp_point(cx+dx,cy+dy,w,h)
                    for idx,(pt) in enumerate(p1g.reshape(-1,2)):
                        if idx < len(tr["trails"]):
                            tr["trails"][idx].append((int(pt[0]),int(pt[1])))
                    p0=p1g.copy()
                    tr["cx"],tr["cy"],tr["p0"]=cx,cy,p0
                    tr["hist"].append((cx,cy))
                    if tr["id"] in REINIT_ACTIVE:
                        del REINIT_ACTIVE[tr["id"]]
                        print(f"[Frame {frame_count}] Tracker {tr['id']} reinit SUCCESS")
            else:
                current_frame = frame_count
                if tr["id"] not in REINIT_ACTIVE:
                    REINIT_ACTIVE[tr["id"]] = current_frame + REINIT_WINDOW
                    print(f"[Frame {frame_count}] Tracker {tr['id']} reinit START: optical flow lost")
                elif current_frame <= REINIT_ACTIVE[tr["id"]]:
                    print(f"[Frame {frame_count}] Tracker {tr['id']} reinit RETRY (optical flow lost)")
                    results = model(frame, classes=[0])
                    detections = []
                    for r in results:
                        for box in r.boxes:
                            x1,y1,x2,y2 = map(int, box.xyxy[0])
                            cx,cy = (x1+x2)/2, (y1+y2)/2
                            p0 = make_support_grid(cx,cy,NEIGH_RADIUS,GRID_SPACING,w,h)
                            if p0 is None:
                                p0=np.array([[[cx,cy]]],dtype=np.float32)
                            detections.append((cx, cy, p0, (x1,y1,x2,y2)))
                    trackers = assign_detections_to_trackers(detections, trackers, frame, video_name, frame_count,ASSIGN_DIST)

                else:
                    print(f"[Frame {frame_count}] Tracker {tr['id']} reinit FAILED (timeout)")
                    del REINIT_ACTIVE[tr["id"]]

        for tr in trackers:
            color = get_id_color(tr["id"])
            cv2.circle(frame,(int(tr["cx"]),int(tr["cy"])),6,color,-1)
            cv2.putText(frame,f"ID {tr['id']}", (int(tr["cx"])+10,int(tr["cy"])-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

            for trail in tr["trails"]:
                if len(trail) > 1:
                    pts = np.array(trail,dtype=np.int32).reshape((-1,1,2))
                    cv2.polylines(frame,[pts],False,color,1)

            if tr["p0"] is not None:
                for (x,y) in tr["p0"].reshape(-1,2):
                    cv2.circle(frame,(int(x),int(y)),3,color,1)

        frame_count+=1
        id_frame_count += 1

        fps=frame_count/(time.perf_counter()-t0+1e-6)
        
        if DASHBOARD_ACTIVE:
            
            dashboard = draw_dashboard_canvas(600, 600, history= VISFRAME, trackers=trackers)

            merged = merge_frame_and_dashboard(frame, dashboard)

            cv2.putText(merged,f"Frame: {id_frame_count} | FPS {fps:.2f} | Trackers={len(trackers)}",(20,30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,TEXT_COLOR,2)

            h, w = merged.shape[:2]

            cv2.putText(merged,f"Frame: {id_frame_count} | FPS {fps:.2f} | Trackers={len(trackers)}",(20,30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,TEXT_COLOR,2)

            cv2.putText(merged, "press D - reset all trackers, clear IDs",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
            cv2.putText(merged, "press I - manual initialization",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
            
            cv2.putText(merged, f"press S - {save_state}",
                        (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)

            
            if SAVE_DATA and last_saved != "No data saved":
                cv2.putText(merged, f"Autosave completed at frame {last_saved}",
                        (800, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
            
            cv2.putText(merged, f"press B - {dashboard_state}",
                        (400, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
            
            labels = [
                "Frame",
                "Tracked IDs",
                "Tracked points"
            ]

            col_width = w // 3
            bar_height = 60

            for i, text in enumerate(labels):
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                text_w, text_h = text_size

                x = i * col_width + (col_width - text_w) // 2
                y = 95

                cv2.putText(merged, text, (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            
            cv2.imshow("Dashboard", merged)
        else:
            cv2.putText(frame,f"Frame: {id_frame_count} | FPS {fps:.2f} | Trackers={len(trackers)}",(20,30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,TEXT_COLOR,2)
            cv2.putText(frame, f"press S - {save_state}",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
            cv2.putText(frame, f"press B - {dashboard_state}",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
            cv2.imshow("Tracker", frame)
        
        key=cv2.waitKey(1)&0xFF
        if key==27:  # ESC -> Quit
            break
        elif key==ord('i') or key==ord('I'):  # "I" key → reinit
            print(f"[Frame {frame_count}] Manual reinitialization")
            used_trackers = set()
            used_detections = set()
            max_dist=100
            for i, tr in enumerate(trackers):
                best_j = -1
                best_d = 1e9
                for j, (cx, cy, p0, box) in enumerate(detections):
                    if j in used_detections: 
                        continue
                    d = np.hypot(tr["cx"] - cx, tr["cy"] - cy)
                    if d < best_d and d < max_dist:
                        best_d = d
                        best_j = j
                if best_j >= 0:
                    cx, cy, p0, box= detections[best_j]
                    tr["cx"], tr["cy"], tr["p0"] = cx, cy, p0
                    tr["init_pts"] = len(p0)
                    tr["trails"] = [deque(maxlen=TRAIL_LEN) for _ in range(len(p0))]
                    tr["hist"].append((cx, cy))
 
                    used_trackers.add(i)
                    used_detections.add(best_j)
 
            save_conflict_frame(frame, video_name, frame_count, trackers, detections, used_detections)
            results = model(frame, classes=[0])
            detections = []
            for r in results:
                for box in r.boxes:
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    cx,cy = (x1+x2)/2, (y1+y2)/2
                    p0 = make_support_grid(cx,cy,NEIGH_RADIUS,GRID_SPACING,w,h)
                    if p0 is None:
                        p0 = np.array([[[cx,cy]]],dtype=np.float32)
                    detections.append((cx, cy, p0, (x1,y1,x2,y2)))
  
            trackers = assign_detections_to_trackers(detections, trackers, frame, video_name, frame_count, ASSIGN_DIST, manual=True)

        elif key==ord('d') or key==ord('D'):  # "D" key → reinit and clear
                    if SAVE_DATA: save_tracks_to_excel_async()
                    track_buffer.clear()
                    print(f"[Frame {frame_count}] All trackers cleared, IDs reset")
                    trackers.clear()
                    REINIT_ACTIVE.clear()
                    ID_COLORS.clear()
                    NEXT_ID = 0
                    id_frame_count = 0
                    current_filename = get_filename()
                    dashboard_data["positions"].clear()
                    dashboard_data["points"].clear()
                    dashboard_data["reinits"].clear()
                    dashboard_data["frames"].clear()

        elif key == ord('s') or key==ord('S'):
            SAVE_DATA = not SAVE_DATA
            print(f"SAVE_DATA = {SAVE_DATA}")
            if SAVE_DATA:
                save_state = "Deactivate saving data"
            else:
                save_state = "Activate saving data"
              
        elif key == ord('b') or key==ord('B'): # "B" key → open/close dashboard
            DASHBOARD_ACTIVE = not DASHBOARD_ACTIVE
            print(f"DASHBOARD_ACTIVE = {DASHBOARD_ACTIVE}")
            if DASHBOARD_ACTIVE:
                cv2.destroyWindow("Tracker")
                dashboard_state = "Deactivate dashboard"
            else:
                cv2.destroyWindow("Dashboard")
                dashboard_state = "Activate dashboard"
                dashboard_data["positions"].clear()
                dashboard_data["points"].clear()
                dashboard_data["reinits"].clear()
                dashboard_data["frames"].clear()
            
            
            
        prev_gray=gray.copy()
    if SAVE_DATA: save_tracks_to_excel_async()
    cap.release(); cv2.destroyAllWindows()

if __name__=="__main__":
    main()
