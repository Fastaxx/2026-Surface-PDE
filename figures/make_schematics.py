#!/usr/bin/env python3
"""
make_schematics.py — generate the DEC schematics used in the deck.

Outputs (next to this script):
    schem_simplices.svg     — 0/1/2-simplices labelled
    schem_dual_area.svg     — primal triangle fan around a vertex + Voronoi dual cell
    schem_hodge.svg         — primal edge ↔ dual edge across two triangles
    schem_laplacian.svg     — cotangent stencil  (L u)_i  ∝ Σ (cot α + cot β)(u_j − u_i)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Circle
from matplotlib.collections import LineCollection

OUT = Path(__file__).parent

def _setup(figsize=(4.2, 3.2)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_facecolor('white'); fig.patch.set_facecolor('white')
    return fig, ax

def _save(fig, name):
    fig.tight_layout(pad=0.4)
    p = OUT / name
    fig.savefig(p, format='svg', bbox_inches='tight', pad_inches=0.05,
                facecolor='white')
    plt.close(fig)
    print('saved', p)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Simplicial complex — vertex (0-simplex), edge (1-simplex), face (2-simplex)
# ─────────────────────────────────────────────────────────────────────────────
def schem_simplices():
    fig, ax = _setup(figsize=(8.0, 3.0))

    # vertex
    ax.scatter([0.5], [0.5], s=140, color='#333', zorder=3)
    ax.text(0.5, 0.05, '0-simplex\n(vertex)', ha='center', fontsize=12)
    ax.text(0.5, 0.85, r'$v_i$', ha='center', fontsize=14)

    # edge
    e = [(2.0, 0.35), (3.2, 0.75)]
    ax.plot([e[0][0], e[1][0]], [e[0][1], e[1][1]],
            color='#1f4e79', lw=3, zorder=2)
    ax.scatter(*zip(*e), s=120, color='#333', zorder=3)
    ax.text(2.6, 0.05, '1-simplex\n(edge)', ha='center', fontsize=12)
    ax.text(2.0 - 0.05, 0.35 - 0.18, r'$v_i$', fontsize=13)
    ax.text(3.2 + 0.03, 0.75 + 0.03, r'$v_j$', fontsize=13)
    ax.text(2.6, 0.9, r'$e_{ij} = [v_i, v_j]$', ha='center', fontsize=12,
            color='#1f4e79')

    # face
    f = np.array([(4.7, 0.30), (5.9, 0.30), (5.3, 1.1)])
    ax.add_patch(Polygon(f, closed=True, facecolor='#fde2c5',
                         edgecolor='#1f4e79', lw=2.5, zorder=1))
    ax.scatter(f[:, 0], f[:, 1], s=120, color='#333', zorder=3)
    ax.text(5.3, 0.05, '2-simplex\n(triangle)', ha='center', fontsize=12)
    ax.text(4.6 - 0.03, 0.30 - 0.18, r'$v_i$', fontsize=13)
    ax.text(5.9 + 0.03, 0.30 - 0.18, r'$v_j$', fontsize=13)
    ax.text(5.3, 1.18, r'$v_k$', ha='center', fontsize=13)
    ax.text(5.3, 1.4, r'$f_{ijk}=[v_i,v_j,v_k]$', ha='center', fontsize=12,
            color='#1f4e79')

    ax.set_xlim(-0.2, 6.5); ax.set_ylim(-0.4, 1.7)
    _save(fig, 'schem_simplices.svg')

# ─────────────────────────────────────────────────────────────────────────────
# 2. Primal mesh + Voronoi dual area around vertex i
# ─────────────────────────────────────────────────────────────────────────────
def schem_dual_area():
    fig, ax = _setup(figsize=(5.0, 4.4))

    # build a fan of triangles around the central vertex v_i
    n_neighbours = 6
    R = 1.6
    theta = np.linspace(0, 2*np.pi, n_neighbours + 1)[:-1]
    # slight irregularity so circumcenters scatter realistically
    rng = np.random.default_rng(7)
    radii = R + 0.15 * rng.standard_normal(n_neighbours)
    ring = np.column_stack([radii * np.cos(theta), radii * np.sin(theta)])
    vi = np.array([0.0, 0.0])

    # primal triangles
    for k in range(n_neighbours):
        tri = np.vstack([vi, ring[k], ring[(k+1) % n_neighbours]])
        ax.add_patch(Polygon(tri, closed=True, facecolor='#f0f0f0',
                             edgecolor='#888', lw=1.0, zorder=1))

    # Voronoi-dual cell: walk circumcenters of each adjacent triangle,
    # connected through edge midpoints (the "circumcentric dual" used in DEC).
    def circumcenter(a, b, c):
        ax1, ay = a; bx, by = b; cx, cy = c
        d = 2 * (ax1*(by - cy) + bx*(cy - ay) + cx*(ay - by))
        ux = ((ax1**2+ay**2)*(by-cy)+(bx**2+by**2)*(cy-ay)+(cx**2+cy**2)*(ay-by))/d
        uy = ((ax1**2+ay**2)*(cx-bx)+(bx**2+by**2)*(ax1-cx)+(cx**2+cy**2)*(bx-ax1))/d
        return np.array([ux, uy])

    dual_pts = []
    for k in range(n_neighbours):
        vj = ring[k]
        vk = ring[(k+1) % n_neighbours]
        mij = 0.5*(vi + vj)
        c   = circumcenter(vi, vj, vk)
        dual_pts.append(mij)
        dual_pts.append(c)
    dual_pts = np.array(dual_pts)

    ax.add_patch(Polygon(dual_pts, closed=True, facecolor='#cfe5ff',
                         edgecolor='#1f4e79', lw=2.2, alpha=0.7, zorder=2))

    # primal edges emphasised
    for v in ring:
        ax.plot([vi[0], v[0]], [vi[1], v[1]], color='#444', lw=1.2, zorder=3)

    # vertices
    ax.scatter(ring[:, 0], ring[:, 1], s=42, color='#333', zorder=4)
    ax.scatter([vi[0]], [vi[1]], s=110, color='#c0392b', zorder=5)

    ax.text(0.06, -0.16, r'$v_i$', fontsize=15, color='#c0392b')
    ax.text(1.1, 1.45,
            r'Voronoi dual cell  $A_i^{\mathrm{dual}}$',
            fontsize=12, color='#1f4e79')
    ax.text(-1.0, -2.05,
            r'Primal: triangle fan around $v_i$  ·  Dual: Voronoi cell',
            fontsize=11, color='#444')

    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
    _save(fig, 'schem_dual_area.svg')

# ─────────────────────────────────────────────────────────────────────────────
# 3. Hodge star: a primal edge crosses its dual edge perpendicularly
# ─────────────────────────────────────────────────────────────────────────────
def schem_hodge():
    fig, ax = _setup(figsize=(5.4, 4.0))

    # two adjacent triangles sharing edge e=(v_i,v_j); tall isoceles so the
    # circumcenters are well-separated and the dual edge is clearly visible.
    vi = np.array([-1.2, 0.0])
    vj = np.array([ 1.2, 0.0])
    vk = np.array([ 0.0,  2.0])         # above
    vl = np.array([ 0.0, -2.0])         # below

    for tri, col in [((vi, vj, vk), '#fde2c5'), ((vi, vj, vl), '#e3d7f4')]:
        ax.add_patch(Polygon(tri, closed=True, facecolor=col,
                             edgecolor='#888', lw=1.5, zorder=1))

    # circumcenters → dual edge e*
    def circumcenter(a, b, c):
        ax1, ay = a; bx, by = b; cx, cy = c
        d = 2 * (ax1*(by - cy) + bx*(cy - ay) + cx*(ay - by))
        ux = ((ax1**2+ay**2)*(by-cy)+(bx**2+by**2)*(cy-ay)+(cx**2+cy**2)*(ay-by))/d
        uy = ((ax1**2+ay**2)*(cx-bx)+(bx**2+by**2)*(ax1-cx)+(cx**2+cy**2)*(bx-ax1))/d
        return np.array([ux, uy])

    c1 = circumcenter(vi, vj, vk)
    c2 = circumcenter(vi, vj, vl)
    ax.plot([c1[0], c2[0]], [c1[1], c2[1]], color='#c0392b', lw=3.0,
            linestyle='--', zorder=3, label=r'$e^*$')
    ax.scatter([c1[0], c2[0]], [c1[1], c2[1]], s=55, color='#c0392b', zorder=4)
    ax.text(c1[0]+0.10, c1[1]+0.08, r'$e^*$  (dual edge)',
            color='#c0392b', fontsize=12)

    # primal edge e on top so it overlays
    ax.plot([vi[0], vj[0]], [vi[1], vj[1]], color='#1f4e79', lw=3.2, zorder=5)
    ax.text(-1.55, 0.18, r'$e$  (primal edge, length $\ell_e$)',
            ha='left', fontsize=12, color='#1f4e79')

    # interior angles α at v_k and β at v_l (opposite the edge e)
    ax.text(vk[0]-0.05, vk[1]-0.55, r'$\alpha_e$',
            fontsize=13, color='#7a4f00', ha='center')
    ax.text(vl[0]-0.05, vl[1]+0.40, r'$\beta_e$',
            fontsize=13, color='#7a4f00', ha='center')

    # vertex dots and labels (placed outside the triangles)
    for p, name, off in [(vi, 'v_i', (-0.20,  0.0)),
                          (vj, 'v_j', ( 0.20,  0.0)),
                          (vk, 'v_k', ( 0.0,   0.22)),
                          (vl, 'v_l', ( 0.0,  -0.28))]:
        ax.scatter([p[0]], [p[1]], s=55, color='#333', zorder=6)
        ax.text(p[0]+off[0], p[1]+off[1], f'${name}$',
                ha='center', va='center', fontsize=13)

    ax.text(0.0, -2.80,
            r'$(\star_1)_{ee} = \dfrac{\ell_e^*}{\ell_e} = \dfrac{\cot\alpha_e + \cot\beta_e}{2}$',
            ha='center', fontsize=13)

    ax.set_xlim(-2.0, 2.0); ax.set_ylim(-3.2, 2.6)
    _save(fig, 'schem_hodge.svg')

# ─────────────────────────────────────────────────────────────────────────────
# 4. Cotangent Laplacian stencil at vertex i
# ─────────────────────────────────────────────────────────────────────────────
def schem_laplacian():
    fig, ax = _setup(figsize=(5.4, 4.4))

    # vertex i in centre, one-ring of 6 neighbours
    n = 6
    R = 1.6
    th = np.linspace(0, 2*np.pi, n + 1)[:-1]
    rng = np.random.default_rng(3)
    rs  = R + 0.12 * rng.standard_normal(n)
    ring = np.column_stack([rs*np.cos(th), rs*np.sin(th)])
    vi = np.array([0.0, 0.0])

    # triangles + edges
    for k in range(n):
        a, b = ring[k], ring[(k+1) % n]
        ax.add_patch(Polygon([vi, a, b], closed=True,
                             facecolor='#f3f3f3', edgecolor='#aaa',
                             lw=1.0, zorder=1))

    # highlight one edge (i, j) and the two opposite angles α, β
    j = 1
    vj = ring[j]
    vk_above = ring[(j-1) % n]    # alpha angle vertex
    vk_below = ring[(j+1) % n]    # beta  angle vertex
    ax.plot([vi[0], vj[0]], [vi[1], vj[1]], color='#1f4e79', lw=3.2, zorder=4)

    # mark the *interior* angle (the small one) at vertex p formed by pq and pr
    def angle_label(p, q, r, txt, color):
        v1 = (q - p); v2 = (r - p)
        a1 = np.arctan2(v1[1], v1[0]); a2 = np.arctan2(v2[1], v2[0])
        # pick the shorter sweep from a1 to a2
        diff = (a2 - a1 + np.pi) % (2*np.pi) - np.pi
        arc = np.linspace(a1, a1 + diff, 30)
        rad = 0.28
        ax.plot(p[0] + rad*np.cos(arc), p[1] + rad*np.sin(arc),
                color=color, lw=1.5)
        mid = a1 + diff/2
        ax.text(p[0] + 0.50*np.cos(mid), p[1] + 0.50*np.sin(mid),
                txt, color=color, fontsize=13, ha='center', va='center')

    angle_label(vk_above, vi, vj, r'$\alpha_{ij}$', '#7a4f00')
    angle_label(vk_below, vi, vj, r'$\beta_{ij}$',  '#7a4f00')

    # vertices
    ax.scatter(ring[:, 0], ring[:, 1], s=46, color='#333', zorder=5)
    ax.scatter([vi[0]], [vi[1]], s=120, color='#c0392b', zorder=6)
    ax.text(0.08, -0.16, r'$v_i$', fontsize=14, color='#c0392b')
    ax.text(vj[0]+0.10, vj[1]+0.10, r'$v_j$', fontsize=13)

    ax.text(0, -2.35,
            r'$(L\,u)_i = \dfrac{1}{A_i^{\mathrm{dual}}}\sum_{j\sim i}'
            r'\dfrac{\cot\alpha_{ij}+\cot\beta_{ij}}{2}\,(u_j-u_i)$',
            ha='center', fontsize=12.5)
    ax.text(0, 2.05,
            r'Surface Laplacian — cotangent stencil at $v_i$',
            ha='center', fontsize=12.5, color='#1f4e79')

    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.7, 2.4)
    _save(fig, 'schem_laplacian.svg')

if __name__ == '__main__':
    schem_simplices()
    schem_dual_area()
    schem_hodge()
    schem_laplacian()
