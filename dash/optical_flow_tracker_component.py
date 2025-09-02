from datetime import datetime
import os
import time


def render_optical_flow_tracker():

    import streamlit as st
    import numpy as np
    import cv2
    import pandas as pd
    import ast

    import People_Tracker_with_Optical_Flow as PT
    st.header("People Tracker – Streamlit integration")
    # ---------------- Controls & Settings ABOVE dashboard ----------------
    ctrl = st.container()
    with ctrl:
        st.subheader("Settings")

        c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
        btn_manual_init = c1.button("Manual reinit")
        btn_reset = c2.button("Reset (clear all)")
        btn_toggle_save = c3.button("Save on/off")
        btn_toggle_dash = c4.button("Dashboard on/off")
        run = c5.toggle("Run", value=st.session_state.get("run", False), key="run")

        st.divider()

        # Source selection
        col_src1, col_src2 = st.columns([1, 2])
        src_mode = col_src1.radio("Source", ["Webcam", "Video File"],
                                  index=st.session_state.get("src_mode_idx", 1), key="src_mode")
        vf = None
        if st.session_state["src_mode"] == "Video File":
            vf = col_src2.file_uploader("Video (mp4/avi/mkv/mov)",
                                        type=["mp4", "avi", "mkv", "mov"], key="video_file")

        st.divider()

        # param.xlsx + sliders
        st.markdown("**Paraméterek** (alapértékek; fájl betöltése után a csúszkák frissülnek)")
        uploaded_param = st.file_uploader("param.xlsx (két oszlop: param, value)",
                                          type=["xlsx"], key="param_xlsx")

        def _apply_params_to_module(param_dict: dict):
            try:
                if hasattr(PT, "params") and isinstance(PT.params, dict):
                    PT.params.update(param_dict)
                else:
                    PT.params = dict(param_dict)
            except Exception:
                PT.params = dict(param_dict)

            for k, v in param_dict.items():
                if k == "text_color" and hasattr(PT, "TEXT_COLOR"):
                    try:
                        setattr(PT, "TEXT_COLOR", tuple(v) if isinstance(v, (list, tuple)) else v)
                        continue
                    except Exception:
                        pass
                UPPER = k.upper()
                if hasattr(PT, UPPER):
                    try:
                        setattr(PT, UPPER, v)
                    except Exception:
                        pass

            if hasattr(PT, "WIN") and hasattr(PT, "LEVELS"):
                try:
                    PT.lk_params = dict(
                        winSize=(int(PT.WIN), int(PT.WIN)),
                        maxLevel=int(PT.LEVELS),
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1, 1e-10),
                    )
                except Exception:
                    pass

        # Parse and apply uploaded param.xlsx (if any), then seed slider defaults
        if uploaded_param is not None:
            try:
                dfp = pd.read_excel(uploaded_param)
                raw = {row["param"]: row["value"] for _, row in dfp.iterrows()}
                for k, v in list(raw.items()):
                    if isinstance(v, str):
                        try:
                            raw[k] = ast.literal_eval(v)
                        except Exception:
                            pass
                _apply_params_to_module(raw)
                # Reflect into sliders
                for k, v in raw.items():
                    st.session_state[f"param_{k}"] = v
                st.success("param.xlsx beolvasva és alkalmazva.")
            except Exception as e:
                st.error(f"param.xlsx beolvasási hiba: {e}")

        # Slider spec (common params)
        param_spec = {
            "MEAN_DRIFT_THRESH": ("Mean drift threshold (px)", 0.0, 200.0, 0.5, float),
            "POINT_DROP_RATIO": ("Point drop ratio", 0.0, 1.0, 0.01, float),
            "ASSIGN_DIST": ("Assign distance (px)", 0.0, 300.0, 1.0, float),
            "REINIT_WINDOW": ("Reinit window (frames)", 1, 240, 1, int),
            "VISFRAME": ("Dashboard history (frames)", 50, 1000, 10, int),
            "N_SAVE_FRAME": ("Autosave period (frames)", 1, 1000, 1, int),
            "WIN": ("LK window (px)", 5, 41, 2, int),
            "LEVELS": ("LK pyramid levels", 0, 3, 1, int),
            "GRID_SPACING": ("Support grid spacing (px)", 2, 32, 1, int),
            "NEIGH_RADIUS": ("Support grid radius (px)", 2, 64, 1, int),
        }

        def _get_current_value(name, cast):
            if hasattr(PT, name):
                val = getattr(PT, name)
            elif hasattr(PT, "params") and isinstance(PT.params, dict) and name.lower() in PT.params:
                val = PT.params[name.lower()]
            else:
                # fallback midpoint
                lo, hi = param_spec[name][1], param_spec[name][2]
                val = 0.5 * (lo + hi)
            try:
                return cast(val)
            except Exception:
                lo, hi = param_spec[name][1], param_spec[name][2]
                return cast(0.5 * (lo + hi))

        sc1, sc2 = st.columns(2)
        updated_params = {}
        for i, (name, (label, vmin, vmax, step, typ)) in enumerate(param_spec.items()):
            default_val = st.session_state.get(f"param_{name}", _get_current_value(name, typ))
            col = sc1 if (i % 2 == 0) else sc2
            if typ is float:
                val = col.slider(label, vmin, vmax, float(default_val), step, key=f"param_{name}")
            else:
                val = col.slider(label, vmin, vmax, int(default_val), step, key=f"param_{name}")
            updated_params[name] = val

        # push back to PT and PT.params
        for name, val in updated_params.items():
            try:
                setattr(PT, name, val)
            except Exception:
                pass
            try:
                if not hasattr(PT, "params") or not isinstance(PT.params, dict):
                    PT.params = {}
                PT.params[name.lower()] = val
            except Exception:
                pass

        st.divider()

    # ---------------- YOLO model cache ----------------
    @st.cache_resource
    def get_model():
        from ultralytics import YOLO
        return YOLO("yolov8n.pt")

    model = get_model()

    # ---------------- State ----------------
    if "initialized" not in st.session_state:
        st.session_state["initialized"] = True
        st.session_state["trackers"] = []
        st.session_state["prev_gray"] = None
        st.session_state["frame_count"] = 0
        st.session_state["video_name"] = None

        # PT globals reset
        if hasattr(PT, "REINIT_ACTIVE"):
            PT.REINIT_ACTIVE.clear()
        if hasattr(PT, "ID_COLORS"):
            PT.ID_COLORS.clear()
        if hasattr(PT, "NEXT_ID"):
            PT.NEXT_ID = 0
        if hasattr(PT, "id_frame_count"):
            PT.id_frame_count = 0
        if hasattr(PT, "track_buffer"):
            PT.track_buffer.clear()
        if hasattr(PT, "current_filename"):
            PT.current_filename = None
        if hasattr(PT, "dashboard_data"):
            PT.dashboard_data["positions"].clear()
            PT.dashboard_data["points"].clear()
            PT.dashboard_data["reinits"].clear()
            PT.dashboard_data["frames"].clear()

    # ---------------- Open source ----------------
    cap = None
    if run:
        if st.session_state["src_mode"] == "Webkamera":
            cap = cv2.VideoCapture(0)
            st.session_state["video_name"] = "camera0"
        else:
            if vf is not None:
                path = os.path.join("/mnt/data", vf.name)
                with open(path, "wb") as f:
                    f.write(vf.read())
                cap = cv2.VideoCapture(path)
                st.session_state["video_name"] = os.path.splitext(os.path.basename(path))[0]

    # ---------------- Layout ----------------
    col_main, col_side = st.columns([3, 1])
    img_slot = col_main.empty()
    meta_box = col_side.container()

    # ---------------- YOLO detection (iou=0.4) ----------------
    def run_yolo_and_build_detections(frame, w, h):
        results = model(frame, classes=[0], iou=0.4, verbose=True)
        detections = []
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                p0 = PT.make_support_grid(cx, cy, PT.NEIGH_RADIUS, PT.GRID_SPACING, w, h)
                if p0 is None:
                    p0 = np.array([[[cx, cy]]], dtype=np.float32)
                detections.append((cx, cy, p0, (x1, y1, x2, y2)))
        return detections

    # ---------------- RESET (D) ----------------
    if btn_reset:
        if getattr(PT, "SAVE_DATA", False) and hasattr(PT, "save_tracks_to_excel_async"):
            PT.save_tracks_to_excel_async()
        if hasattr(PT, "track_buffer"):
            PT.track_buffer.clear()
        st.session_state["trackers"].clear()
        if hasattr(PT, "REINIT_ACTIVE"):
            PT.REINIT_ACTIVE.clear()
        if hasattr(PT, "ID_COLORS"):
            PT.ID_COLORS.clear()
        if hasattr(PT, "NEXT_ID"):
            PT.NEXT_ID = 0
        if hasattr(PT, "id_frame_count"):
            PT.id_frame_count = 0
        if hasattr(PT, "current_filename"):
            PT.current_filename = PT.get_filename() if hasattr(PT, "get_filename") else None
        if hasattr(PT, "dashboard_data"):
            PT.dashboard_data["positions"].clear()
            PT.dashboard_data["points"].clear()
            PT.dashboard_data["reinits"].clear()
            PT.dashboard_data["frames"].clear()
        st.session_state["prev_gray"] = None
        st.session_state["frame_count"] = 0
        st.success("RESET: trackers reset, IDs and buffers cleared.")

    # ---------------- SAVE toggle (S) ----------------
    if btn_toggle_save and hasattr(PT, "SAVE_DATA"):
        PT.SAVE_DATA = not PT.SAVE_DATA
        if hasattr(PT, "save_state"):
            PT.save_state = "Deactivate saving data" if PT.SAVE_DATA else "Activate saving data"
        st.info(f"SAVE_DATA = {PT.SAVE_DATA}")

    # ---------------- DASHBOARD toggle (B) ----------------
    if btn_toggle_dash and hasattr(PT, "DASHBOARD_ACTIVE"):
        PT.DASHBOARD_ACTIVE = not PT.DASHBOARD_ACTIVE
        if hasattr(PT, "dashboard_state"):
            PT.dashboard_state = "Deactivate dashboard" if PT.DASHBOARD_ACTIVE else "Activate dashboard"
        if (not PT.DASHBOARD_ACTIVE) and hasattr(PT, "dashboard_data"):
            PT.dashboard_data["positions"].clear()
            PT.dashboard_data["points"].clear()
            PT.dashboard_data["reinits"].clear()
            PT.dashboard_data["frames"].clear()
        st.info(f"DASHBOARD_ACTIVE = {PT.DASHBOARD_ACTIVE}")

    # ---------------- Return wrapper: unify return type ----------------
    def _assign_only_list(ret):
        try:
            if isinstance(ret, tuple) and len(ret) >= 1:
                return ret[0]
        except Exception:
            pass
        return ret

    # ---------------- Main loop ----------------
    if run and cap is not None and cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            st.error("Source not readable.")
            cap.release()
        else:
            h, w = frame.shape[:2]
            st.session_state["prev_gray"] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            st.session_state["frame_count"] = 0
            if hasattr(PT, "id_frame_count"):
                PT.id_frame_count = 0
            video_name = st.session_state["video_name"] or "video"
            t0 = time.perf_counter()

            while True:
                ok, frame = cap.read()
                if not ok:
                    st.info("No more frames.")
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Log buffer
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                for tr in st.session_state["trackers"]:
                    if hasattr(PT, "track_buffer"):
                        PT.track_buffer.append([getattr(PT, "id_frame_count", 0), timestamp, tr["id"], tr["cx"], tr["cy"]])

                if getattr(PT, "SAVE_DATA", False) and hasattr(PT, "N_SAVE_FRAME") and hasattr(PT, "save_tracks_to_excel_async"):
                    if getattr(PT, "id_frame_count", 0) % max(1, int(PT.N_SAVE_FRAME)) == 0 and getattr(PT, "id_frame_count", 0) > 0:
                        PT.save_tracks_to_excel_async()

                if getattr(PT, "DASHBOARD_ACTIVE", True) and hasattr(PT, "dashboard_data"):
                    PT.dashboard_data["frames"].append(getattr(PT, "id_frame_count", 0))
                    for tr in st.session_state["trackers"]:
                        tid = tr["id"]
                        PT.dashboard_data["positions"][tid]["x"].append(tr["cx"])
                        PT.dashboard_data["positions"][tid]["y"].append(tr["cy"])
                        PT.dashboard_data["points"][tid].append(len(tr["p0"]))
                        if hasattr(PT, "REINIT_ACTIVE") and tid in PT.REINIT_ACTIVE:
                            PT.dashboard_data["reinits"][tid].append(getattr(PT, "id_frame_count", 0))

                # Initial detection (if no trackers exist)
                if len(st.session_state["trackers"]) == 0:
                    detections = run_yolo_and_build_detections(frame, w, h)
                    st.session_state["trackers"] = _assign_only_list(PT.assign_detections_to_trackers(
                        detections, st.session_state["trackers"], frame, video_name, st.session_state["frame_count"], PT.ASSIGN_DIST
                    ))

                # LK criteria update + reinit requests collection
                updated = []
                need_reinit_ids = set()

                for tr in st.session_state["trackers"]:
                    cx, cy, p0 = tr["cx"], tr["cy"], tr["p0"]
                    p1g, p0g = PT.forward_backward_filter(st.session_state["prev_gray"], gray, p0)

                    if (p1g is not None) and (len(p1g) > 0):
                        pts = p1g.reshape(-1, 2)
                        dists = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
                        k = max(1, int(len(dists) * 0.1))
                        worst_mean = float(np.mean(np.sort(dists)[-k:]))

                        init_pts = int(tr.get("init_pts", len(p1g)))
                        init_pts = max(1, init_pts)
                        too_few = (len(p1g) < PT.POINT_DROP_RATIO * init_pts)

                        if too_few or (worst_mean > PT.MEAN_DRIFT_THRESH):
                            need_reinit_ids.add(tr["id"])
                            current_frame = st.session_state["frame_count"]
                            reason = None
                            # DRIFT first (to match logs like "mean drift too large =...px)")
                            if worst_mean > PT.MEAN_DRIFT_THRESH:
                                reason = f"mean drift too large ={worst_mean:.1f}px)"
                            else:
                                reason = f"too few points: ({len(p1g)}/{init_pts})"
                            if hasattr(PT, "REINIT_ACTIVE"):
                                if tr["id"] not in PT.REINIT_ACTIVE:
                                    PT.REINIT_ACTIVE[tr["id"]] = current_frame + PT.REINIT_WINDOW
                                    print(f"[Frame {st.session_state['frame_count']}] Tracker {tr['id']} reinit START: {reason}")
                                else:
                                    if current_frame <= PT.REINIT_ACTIVE[tr["id"]]:
                                        print(f"[Frame {st.session_state['frame_count']}] Tracker {tr['id']} retrying reinit ({reason})")
                                    else:
                                        print(f"[Frame {st.session_state['frame_count']}] Tracker {tr['id']} reinit FAILED (timeout)")
                                        del PT.REINIT_ACTIVE[tr["id"]]
                        else:
                            # normal update
                            disp = (p1g - p0g).reshape(-1, 2)
                            dx, dy = float(np.mean(disp[:, 0])), float(np.mean(disp[:, 1]))
                            cx, cy = PT.clamp_point(cx + dx, cy + dy, w, h)
                            for idx, pt in enumerate(p1g.reshape(-1, 2)):
                                if idx < len(tr["trails"]):
                                    tr["trails"][idx].append((int(pt[0]), int(pt[1])))
                            tr["cx"], tr["cy"], tr["p0"] = cx, cy, p1g.copy()
                            tr["hist"].append((cx, cy))
                            if hasattr(PT, "REINIT_ACTIVE") and tr["id"] in PT.REINIT_ACTIVE:
                                del PT.REINIT_ACTIVE[tr["id"]]
                                print(f"[Frame {st.session_state['frame_count']}] Tracker {tr['id']} reinit SUCCESS")
                            updated.append(tr)
                    else:
                        # optical flow lost
                        need_reinit_ids.add(tr["id"])
                        current_frame = st.session_state["frame_count"]
                        if hasattr(PT, "REINIT_ACTIVE"):
                            if tr["id"] not in PT.REINIT_ACTIVE:
                                PT.REINIT_ACTIVE[tr["id"]] = current_frame + PT.REINIT_WINDOW
                                print(f"[Frame {st.session_state['frame_count']}] Tracker {tr['id']} reinit START: optical flow lost")
                            else:
                                if current_frame <= PT.REINIT_ACTIVE[tr["id"]]:
                                    print(f"[Frame {st.session_state['frame_count']}] Tracker {tr['id']} reinit RETRY (optical flow lost)")
                                else:
                                    print(f"[Frame {st.session_state['frame_count']}] Tracker {tr['id']} reinit FAILED (timeout)")
                                    del PT.REINIT_ACTIVE[tr["id"]]

                # Single detection+assign at end of frame if any tracker needs reinit and window alive
                if need_reinit_ids and any(st.session_state["frame_count"] <= PT.REINIT_ACTIVE.get(tid, -1) for tid in need_reinit_ids):
                    dets = run_yolo_and_build_detections(frame, w, h)
                    new_list = _assign_only_list(PT.assign_detections_to_trackers(
                        dets, st.session_state["trackers"], frame, video_name, st.session_state["frame_count"], PT.ASSIGN_DIST
                    ))
                    # Success check: enough points?
                    for tr in new_list:
                        if tr["id"] in need_reinit_ids and tr.get("p0") is not None:
                            init_pts2 = int(tr.get("init_pts", len(tr["p0"])))
                            init_pts2 = max(1, init_pts2)
                            if len(tr["p0"]) >= PT.POINT_DROP_RATIO * init_pts2:
                                if tr["id"] in PT.REINIT_ACTIVE:
                                    del PT.REINIT_ACTIVE[tr["id"]]
                                    print(f"[Frame {st.session_state['frame_count']}] Tracker {tr['id']} reinit SUCCESS")
                    st.session_state["trackers"] = new_list
                else:
                    st.session_state["trackers"] = updated if len(updated) > 0 else st.session_state["trackers"]

                # Manual reinitialization (I)
                if btn_manual_init:
                    print(f"[Frame {st.session_state['frame_count']}] Manual reinitialization")
                    detections = run_yolo_and_build_detections(frame, w, h)
                    try:
                        st.session_state["trackers"] = _assign_only_list(PT.assign_detections_to_trackers(
                            detections, st.session_state["trackers"], frame, video_name, st.session_state["frame_count"], PT.ASSIGN_DIST, manual=True
                        ))
                    except TypeError:
                        st.session_state["trackers"] = _assign_only_list(PT.assign_detections_to_trackers(
                            detections, st.session_state["trackers"], frame, video_name, st.session_state["frame_count"], PT.ASSIGN_DIST
                        ))

                # Drawing (overlay + dashboard merge)
                draw = frame.copy()
                for tr in st.session_state["trackers"]:
                    color = PT.get_id_color(tr["id"])
                    cv2.circle(draw, (int(tr["cx"]), int(tr["cy"])), 6, color, -1)
                    cv2.putText(draw, f"ID {tr['id']}", (int(tr["cx"]) + 10, int(tr["cy"]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    for trail in tr["trails"]:
                        if len(trail) > 1:
                            pts = np.array(trail, dtype=np.int32).reshape((-1, 1, 2))
                            cv2.polylines(draw, [pts], False, color, 1)
                    if tr["p0"] is not None:
                        for (x, y) in tr["p0"].reshape(-1, 2):
                            cv2.circle(draw, (int(x), int(y)), 3, color, 1)

                st.session_state["frame_count"] += 1
                if hasattr(PT, "id_frame_count"):
                    PT.id_frame_count += 1
                fps = st.session_state["frame_count"] / (time.perf_counter() - t0 + 1e-6)

                if getattr(PT, "DASHBOARD_ACTIVE", True):
                    dashboard = PT.draw_dashboard_canvas(600, 600, history=getattr(PT, "VISFRAME", 200), trackers=st.session_state["trackers"])
                    merged = PT.merge_frame_and_dashboard(draw, dashboard)
                    cv2.putText(merged, f"Frame: {getattr(PT, 'id_frame_count', st.session_state['frame_count'])} | FPS {fps:.2f} | Trackers={len(st.session_state['trackers'])}", (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, getattr(PT, "TEXT_COLOR", (255, 255, 255)), 2)
                    img_slot.image(cv2.cvtColor(merged, cv2.COLOR_BGR2RGB), use_container_width=True)
                else:
                    cv2.putText(draw, f"Frame: {getattr(PT, 'id_frame_count', st.session_state['frame_count'])} | FPS {fps:.2f} | Trackers={len(st.session_state['trackers'])}", (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, getattr(PT, "TEXT_COLOR", (255, 255, 255)), 2)
                    img_slot.image(cv2.cvtColor(draw, cv2.COLOR_BGR2RGB), use_container_width=True)

                st.session_state["prev_gray"] = gray
                time.sleep(0.001)

            cap.release()
