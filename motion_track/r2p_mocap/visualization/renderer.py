"""
Real-time 3D skeleton renderer using matplotlib.

Draws the R2P humanoid skeleton as connected bone segments with coloured
limb groups.  Foot contact state is reflected in foot-marker colour.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

from ..core.skeleton import Bone, BONE_PARENTS, SkeletonPose
from ..biomechanics.foot_contact import FootState


# ── Bone connection chains (for drawing lines) ────────────────────────────────
_CHAINS: List[Tuple[str, List[Bone]]] = [
    ("spine",   [Bone.PELVIS, Bone.SPINE_1, Bone.SPINE_2, Bone.CHEST, Bone.NECK, Bone.HEAD]),
    ("l_arm",   [Bone.CHEST, Bone.L_SHOULDER, Bone.L_UPPER_ARM, Bone.L_FOREARM, Bone.L_HAND]),
    ("r_arm",   [Bone.CHEST, Bone.R_SHOULDER, Bone.R_UPPER_ARM, Bone.R_FOREARM, Bone.R_HAND]),
    ("l_leg",   [Bone.PELVIS, Bone.L_THIGH, Bone.L_SHIN, Bone.L_FOOT, Bone.L_TOE]),
    ("r_leg",   [Bone.PELVIS, Bone.R_THIGH, Bone.R_SHIN, Bone.R_FOOT, Bone.R_TOE]),
    ("shoulder",[Bone.L_SHOULDER, Bone.CHEST, Bone.R_SHOULDER]),
    ("hips",    [Bone.L_THIGH, Bone.PELVIS, Bone.R_THIGH]),
]

_CHAIN_COLORS: Dict[str, str] = {
    "spine":    "#FFFFFF",
    "l_arm":    "#4FC3F7",
    "r_arm":    "#EF9A9A",
    "l_leg":    "#81C784",
    "r_leg":    "#FFB74D",
    "shoulder": "#CE93D8",
    "hips":     "#CE93D8",
}


class SkeletonRenderer:
    """
    Live matplotlib 3D viewer.

    Usage:
        renderer = SkeletonRenderer(title="R2P Mocap")
        renderer.start()                      # opens the window
        ...
        renderer.update(pose, foot_states)    # call from engine loop
        renderer.show()                       # blocks until window closed
    """

    def __init__(
        self,
        title:      str   = "R2P — Inertial Motion Capture",
        view_range: float = 1.2,
        fps:        int   = 30,
    ):
        self.title = title
        self.view_range = view_range
        self.fps = fps
        self._fig: Optional[plt.Figure] = None
        self._ax:  Optional[Axes3D] = None
        self._lines: Dict[str, plt.Line2D] = {}
        self._texts: Dict[str, plt.Text] = {}
        self._pose: Optional[SkeletonPose] = None
        self._foot_states: Optional[Dict[str, FootState]] = None
        self._motion_label: str = ""
        self._frame_num: int = 0
        self._root_trail: List[np.ndarray] = []
        self._trail_len = 120

    # ------------------------------------------------------------------
    def start(self):
        """Create the matplotlib window (non-blocking)."""
        plt.style.use("dark_background")
        self._fig = plt.figure(figsize=(10, 8), facecolor="#1a1a2e")
        self._ax  = self._fig.add_subplot(111, projection="3d", facecolor="#1a1a2e")
        self._ax.set_title(self.title, color="white", fontsize=14, pad=10)
        self._configure_axes()
        self._init_artists()
        plt.tight_layout()
        plt.ion()
        plt.show(block=False)

    def stop(self):
        plt.close(self._fig)

    # ------------------------------------------------------------------
    def update(
        self,
        pose:         SkeletonPose,
        foot_states:  Optional[Dict[str, FootState]] = None,
        motion_label: str = "",
        redraw:       bool = True,
    ):
        self._pose = pose
        self._foot_states = foot_states
        self._motion_label = motion_label
        self._frame_num += 1

        if pose is None:
            return
        self._root_trail.append(pose.root_position.copy())
        if len(self._root_trail) > self._trail_len:
            self._root_trail.pop(0)

        self._draw()
        if redraw:
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()

    # ------------------------------------------------------------------
    def render_sequence(
        self,
        poses:       List[SkeletonPose],
        foot_seq:    Optional[List[Dict[str, FootState]]] = None,
        labels:      Optional[List[str]] = None,
        interval_ms: int = 10,
    ):
        """Replay a pre-recorded sequence as a matplotlib animation."""
        self.start()

        def _animate(i):
            fs = foot_seq[i] if foot_seq else None
            lb = labels[i]   if labels  else ""
            self.update(poses[i], fs, lb, redraw=False)
            return list(self._lines.values())

        ani = animation.FuncAnimation(
            self._fig, _animate, frames=len(poses),
            interval=interval_ms, blit=False, repeat=True,
        )
        plt.show(block=True)
        return ani

    # ------------------------------------------------------------------
    def _configure_axes(self):
        ax = self._ax
        r  = self.view_range
        ax.set_xlim(-r, r)
        ax.set_ylim(0, r * 2)
        ax.set_zlim(-r, r)
        ax.set_xlabel("X (right)",   color="#888888", labelpad=5)
        ax.set_ylabel("Y (up)",      color="#888888", labelpad=5)
        ax.set_zlabel("Z (forward)", color="#888888", labelpad=5)
        ax.tick_params(colors="#444444", labelsize=7)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("#333333")
        ax.yaxis.pane.set_edgecolor("#333333")
        ax.zaxis.pane.set_edgecolor("#333333")
        ax.grid(True, color="#333333", linewidth=0.5)
        # Ground plane
        gx = np.array([-r, r, r, -r, -r])
        gz = np.array([-r, -r, r, r, -r])
        gy = np.zeros(5)
        ax.plot(gx, gy, gz, color="#2a2a4a", linewidth=1, zorder=0)

    def _init_artists(self):
        ax = self._ax
        for chain_name, _ in _CHAINS:
            color = _CHAIN_COLORS[chain_name]
            (line,) = ax.plot([], [], [], color=color, linewidth=2.5,
                              solid_capstyle="round", zorder=5)
            self._lines[chain_name] = line

        # Trail line
        (line,) = ax.plot([], [], [], color="#555577", linewidth=1,
                          alpha=0.6, zorder=2)
        self._lines["trail"] = line

        # Foot contact markers
        for side in ("l_foot", "r_foot"):
            (line,) = ax.plot([], [], [], "o", markersize=8, zorder=6)
            self._lines[side] = line

        # Info text
        self._texts["info"] = ax.text2D(
            0.02, 0.97, "", transform=ax.transAxes,
            color="white", fontsize=9, verticalalignment="top",
            fontfamily="monospace",
        )

    def _draw(self):
        pose = self._pose
        if pose is None or not pose.bones:
            return

        def pos(b: Bone) -> Optional[np.ndarray]:
            bt = pose.bones.get(b)
            return bt.position if bt else None

        # Bone chains
        for chain_name, chain_bones in _CHAINS:
            xs, ys, zs = [], [], []
            for b in chain_bones:
                p = pos(b)
                if p is not None:
                    xs.append(p[0])
                    ys.append(p[1])
                    zs.append(p[2])
            line = self._lines[chain_name]
            if xs:
                line.set_data_3d(xs, ys, zs)

        # Root trail (XYZ, but matplotlib plot_3d uses x,y,z differently)
        if len(self._root_trail) > 1:
            trail = np.array(self._root_trail)
            self._lines["trail"].set_data_3d(trail[:, 0], trail[:, 1], trail[:, 2])

        # Foot contact indicators
        for side, bone, sid in [("left", Bone.L_FOOT, "l_foot"), ("right", Bone.R_FOOT, "r_foot")]:
            p = pos(bone)
            if p is not None:
                contact = (
                    self._foot_states[side].contact
                    if self._foot_states and side in self._foot_states
                    else True
                )
                color = "#00FF88" if contact else "#FF4444"
                self._lines[sid].set_data_3d([p[0]], [p[1]], [p[2]])
                self._lines[sid].set_color(color)

        # Re-centre view on root
        root = pose.root_position
        r = self.view_range
        self._ax.set_xlim(root[0] - r, root[0] + r)
        self._ax.set_ylim(0, r * 2)
        self._ax.set_zlim(root[2] - r, root[2] + r)

        # Info label
        fs = self._foot_states
        lc = "CONTACT" if (fs and fs.get("left")  and fs["left"].contact)  else "SWING"
        rc = "CONTACT" if (fs and fs.get("right") and fs["right"].contact) else "SWING"
        info = (
            f"Frame:  {self._frame_num:>6}\n"
            f"Motion: {self._motion_label}\n"
            f"Root:   ({root[0]:+.2f}, {root[1]:.2f}, {root[2]:+.2f})\n"
            f"L foot: {lc}\n"
            f"R foot: {rc}"
        )
        self._texts["info"].set_text(info)
