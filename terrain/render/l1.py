"""L1 renderer (DESIGN §10.1): top-down heightmap view, cheap enough to run after every tool call.

Hillshade + elevation tint, water overlay, contours, a 0.1-step normalized (u, v) grid matching tool
coordinates, a north arrow and a 32-block scale bar. Fixed size (default 1024 px) to bound image-token
cost. Needs the `render` extra (matplotlib).
"""

from __future__ import annotations

import io

import numpy as np

from terrain.state import WorldState


def render_l1(state: WorldState, title: str = "", size_px: int = 1024, contour_step: float = 10.0) -> bytes:
    """PNG bytes of a north-up relief map with a 0.1-step (u, v) grid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource, ListedColormap

    h = state.height.astype(np.float64)
    dpi = 128
    fig, ax = plt.subplots(figsize=(size_px / dpi, size_px / dpi), dpi=dpi)
    extent = (0.0, 1.0, 1.0, 0.0)  # u to the east, v to the south, (0, 0) top-left = north-west

    # Land part of the "terrain" colormap only, so low ground is never mistaken for water.
    land = ListedColormap(plt.get_cmap("terrain")(np.linspace(0.25, 1.0, 256)))
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(h, cmap=land, vert_exag=1.0, blend_mode="soft", vmin=float(h.min()) - 1, vmax=float(h.max()) + 1)
    ax.imshow(rgb, extent=extent, interpolation="bilinear")

    wet = ~np.isnan(state.water)
    if wet.any():
        depth = np.where(wet, state.water - h, np.nan)
        deepest = max(float(np.nanmax(depth)), 4.0)
        # vmin below 0 keeps even 1-block-deep water clearly blue.
        ax.imshow(np.ma.masked_invalid(depth), extent=extent, cmap="Blues", vmin=-0.6 * deepest, vmax=deepest, interpolation="nearest")

    if h.max() - h.min() >= contour_step:
        levels = np.arange(np.ceil(h.min() / contour_step) * contour_step, h.max(), contour_step)
        u = np.linspace(0, 1, state.width)
        v = np.linspace(0, 1, state.depth)
        ax.contour(u, v, h, levels=levels, colors="k", linewidths=0.3, alpha=0.5)

    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.grid(color="white", linewidth=0.4, alpha=0.6)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("u (east →)")
    ax.set_ylabel("v (south ↓)")
    ax.set_title(title or f"{state.width}×{state.depth} blocks, y={h.min():.0f}..{h.max():.0f}", fontsize=9)
    # North = v 0 = top of the image: arrow head at the top.
    ax.annotate("N", xy=(0.96, 0.97), xytext=(0.96, 0.87), xycoords="axes fraction", ha="center", va="top",
                arrowprops={"arrowstyle": "->", "color": "black"}, color="black", fontsize=9)
    # Scale bar: 32 blocks, bottom-left.
    bar = 32 / max(state.width - 1, 1)
    ax.plot([0.03, 0.03 + bar], [0.96, 0.96], color="black", linewidth=3, solid_capstyle="butt")
    ax.text(0.03 + bar / 2, 0.945, "32 blocks", ha="center", va="bottom", fontsize=7)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()
