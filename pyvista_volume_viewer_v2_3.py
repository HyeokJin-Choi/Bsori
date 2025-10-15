# -*- coding: utf-8 -*-
# PyVista + Panel 3D Volume Viewer (v2.5, Offscreen Snapshot via separate Plotter)
# - ImageData로 spacing 적용 후 add_volume(grid, scalars="values")
# - Status 패널, 카메라 reset + clipping 보정, 강제 render
# - Orthogonal slices / Isosurface 지원
# - ✅ Snapshot: 별도 off_screen Plotter에서 장면 재구성 → PNG 캡처 (브라우저 WebGL 불필요)

import os, traceback, io
import numpy as np
import panel as pn
import pyvista as pv
from PIL import Image

pn.extension("vtk")

DEFAULT_PATH = "yz_volume_out/volume_zyx.npy"  # (Z,Y,X)

# ---------------- Widgets ----------------
volume_path = pn.widgets.TextInput(
    name="Volume .npy path (Z,Y,X)", value=DEFAULT_PATH, sizing_mode="stretch_width"
)
dx = pn.widgets.FloatInput(name="dx (X spacing)", value=1.0, step=0.1)
dy = pn.widgets.FloatInput(name="dy (Y spacing)", value=1.0, step=0.1)
dz = pn.widgets.FloatInput(name="dz (Z spacing)", value=1.0, step=0.1)
cmap = pn.widgets.Select(
    name="Colormap", value="viridis",
    options=["gray","viridis","plasma","magma","inferno","bone","cividis","turbo"]
)
opacity_preset = pn.widgets.Select(
    name="Opacity preset", value="geom",
    options=["linear","sigmoid","geom","sqrt","cubic"]
)
iso_value = pn.widgets.FloatSlider(name="Iso value", start=0, end=255, step=0.5, value=180)
show_volume = pn.widgets.Checkbox(name="Show Volume", value=True)
show_isosurface = pn.widgets.Checkbox(name="Show Isosurface", value=False)
show_slices = pn.widgets.Checkbox(name="Show Orthogonal Slices", value=True)

x_idx = pn.widgets.IntSlider(name="X index (YZ plane)", start=0, end=5, value=0)
y_idx = pn.widgets.IntSlider(name="Y index (ZX plane)", start=0, end=5, value=0)
z_idx = pn.widgets.IntSlider(name="Z index (XY plane)", start=0, end=5, value=0)

reload_btn = pn.widgets.Button(name="Load / Reload Volume", button_type="primary")
status_md = pn.pane.Markdown("**Status:** Ready. Click 'Load / Reload Volume'.", sizing_mode="stretch_width")

# ---------------- Plotter / State ----------------
plotter = pv.Plotter()  # 메인(Panel용) 플로터
# 수정 후 (방법 1: 높이 반응형)
plot_pane = pn.pane.VTK(plotter.ren_win, sizing_mode="stretch_both")

state = {
    "vol": None,          # numpy (Z,Y,X)
    "grid": None,         # pv.ImageData (X,Y,Z) with 'values'
    "vmin": None, "vmax": None,
    "shape": None,
    "rendered": False
}

def log(msg: str):
    status_md.object = f"**Status:** {msg}"
    print(msg)

def build_imagedata(vol: np.ndarray, spacing):
    """(Z,Y,X) -> pv.ImageData with (X,Y,Z) dims and point scalars 'values'."""
    Z, Y, X = vol.shape
    grid = pv.ImageData()
    grid.dimensions = (X, Y, Z)              # X,Y,Z 순서
    grid.spacing = spacing                   # (dx,dy,dz)
    grid.origin = (0.0, 0.0, 0.0)
    data = np.ascontiguousarray(vol.transpose(2,1,0)).ravel(order="F")
    grid.point_data["values"] = data
    return grid

def load_volume_event(event=None):
    try:
        path = volume_path.value.strip()
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        vol = np.load(path)
        if vol.ndim != 3:
            raise ValueError(f"Volume must be 3D (Z,Y,X). Got shape {vol.shape}")

        spacing = (float(dx.value), float(dy.value), float(dz.value))
        grid = build_imagedata(vol, spacing)

        state["vol"] = vol
        state["grid"] = grid
        state["vmin"], state["vmax"] = float(vol.min()), float(vol.max())
        state["shape"] = vol.shape
        state["rendered"] = False

        # 슬라이더/ISO 범위 업데이트
        Z, Y, X = vol.shape
        x_idx.end = max(0, X-1); x_idx.value = X//2
        y_idx.end = max(0, Y-1); y_idx.value = Y//2
        z_idx.end = max(0, Z-1); z_idx.value = Z//2
        iso_value.start = state["vmin"]; iso_value.end = state["vmax"]; iso_value.value = (state["vmin"]+state["vmax"])/2.0

        log(f"Loaded {path} | shape=(Z={Z},Y={Y},X={X}) min={state['vmin']} max={state['vmax']}")
        refresh_scene()
    except Exception as e:
        log(f"Load error: {e}\n```\n{traceback.format_exc()}\n```")

def _add_content_to_plotter(p: pv.Plotter, vol: np.ndarray, grid: pv.ImageData):
    """현재 상태(show_* 토글/파라미터)에 맞춰 플로터 p에 콘텐츠 추가."""
    # Volume
    if show_volume.value:
        p.add_volume(grid, scalars="values", cmap=cmap.value, opacity=opacity_preset.value, shade=True)

    # Isosurface
    if show_isosurface.value:
        try:
            iso = grid.contour(isosurfaces=[float(iso_value.value)], scalars="values")
            p.add_mesh(iso, color="orange", opacity=0.8)
        except Exception as e:
            log(f"Isosurface error: {e}")

    # Orthogonal slices
    if show_slices.value:
        Z, Y, X = vol.shape
        xi = max(0, min(X-1, int(x_idx.value)))
        yi = max(0, min(Y-1, int(y_idx.value)))
        zi = max(0, min(Z-1, int(z_idx.value)))
        px = xi * float(dx.value); py = yi * float(dy.value); pz = zi * float(dz.value)
        s = grid.slice_orthogonal(x=px, y=py, z=pz, generate_triangles=True)
        p.add_mesh(s, cmap=cmap.value, opacity=1.0, scalar_bar_args={"title": "intensity"})

def refresh_scene(event=None):
    try:
        plotter.clear()
        vol = state["vol"]; grid = state["grid"]
        if vol is None or grid is None:
            log("No volume loaded yet.")
            plotter.render()
            plot_pane.object = plotter.ren_win
            state["rendered"] = False
            return

        _add_content_to_plotter(plotter, vol, grid)

        # 카메라/렌더 보정
        plotter.set_background("black")
        plotter.add_axes(); plotter.show_grid()
        plotter.reset_camera()
        plotter.reset_camera_clipping_range()
        plotter.render()
        plot_pane.object = plotter.ren_win
        state["rendered"] = True
        log("Render updated (camera & clipping reset).")
    except Exception as e:
        state["rendered"] = False
        log(f"Render error: {e}\n```\n{traceback.format_exc()}\n```")

# ✅ Snapshot: 별도 off_screen Plotter에서 장면 재구성 후 캡처
def _snapshot_callback():
    try:
        vol = state["vol"]; grid = state["grid"]
        if vol is None or grid is None:
            raise RuntimeError("No volume loaded to snapshot.")
        # 오프스크린 전용 플로터 생성
        p = pv.Plotter(off_screen=True, window_size=(1280, 960))
        _add_content_to_plotter(p, vol, grid)
        p.set_background("black")
        p.reset_camera(); p.reset_camera_clipping_range()
        img = p.screenshot(return_img=True)  # off_screen 플로터는 키워드 없이 동작
        p.close()

        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        buf.seek(0)
        log("Snapshot captured (off-screen).")
        return buf  # FileDownload가 처리 가능한 file-like 객체
    except Exception as e:
        log(f"Snapshot error: {e}\n```\n{traceback.format_exc()}\n```")
        # 실패 시 빈 PNG 반환
        empty = Image.new("RGB", (640, 480), (0, 0, 0))
        buf = io.BytesIO()
        empty.save(buf, format="PNG")
        buf.seek(0)
        return buf

snapshot_download = pn.widgets.FileDownload(
    label="Download PNG Snapshot",
    filename="volume_render.png",
    button_type="success",
    callback=_snapshot_callback,
)

# 이벤트 바인딩
reload_btn.on_click(load_volume_event)
for w in (dx,dy,dz,cmap,opacity_preset,show_volume,show_isosurface,show_slices,x_idx,y_idx,z_idx,iso_value):
    w.param.watch(refresh_scene, "value")

# ---------------- Layout ----------------
controls_left = pn.Column(
    "### Data & Appearance",
    volume_path, pn.Row(dx, dy, dz), cmap, opacity_preset, iso_value,
    pn.Row(show_volume, show_isosurface, show_slices),
    reload_btn, snapshot_download, status_md, sizing_mode="stretch_width"
)
# controls_left = pn.Column(
#     "### Data & Appearance",
#     volume_path, pn.Row(dx, dy, dz), cmap, opacity_preset, iso_value,
#     pn.Row(show_volume, show_isosurface, show_slices),
#     reload_btn,  # snapshot_download 제거
#     status_md, sizing_mode="stretch_width"
# )

controls_right = pn.Column("### Slice indices", x_idx, y_idx, z_idx, sizing_mode="stretch_width")
layout = pn.Row(pn.Column(controls_left, controls_right, width=420), plot_pane, sizing_mode="stretch_both")

layout.servable(title="PyVista 3D Volume Viewer (v2.5)")
