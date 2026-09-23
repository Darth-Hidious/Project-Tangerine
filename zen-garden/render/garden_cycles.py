"""Photoreal render of the garden twin with Blender Cycles.

Builds the scene from the bundle that experiments/export_render_scene.py writes, so every
height and outline is the model's, and renders it with physically based materials:

  - an oiled walnut frame of four mitred boards with rounded tops, on an oak table;
  - the sand exactly as the simulation left it (0.5 mm grid), with quartz grains and glints;
  - moss as hundreds of thousands of instanced leafy shoots, darker living moss near the water
    and brighter preserved moss beyond, on the ground the model computed;
  - water that refracts over a pebbled bed, cascades as aerated sheets with foam;
  - granite lanterns whose paper windows give out their LEDs' 60 lm at 2200 K in the evening,
    in the model's 15 lux of room light; rocks with moss on their crowns, the two stones;
  - the bonsai grown by karesansui.bonsai: plated bark on a tapering trunk, pine shoots on its twigs;
  - the SCARA arm in the pose it holds at the end of its last planned groove.

Every procedural texture is built here from nodes: nothing is downloaded.

Run (Blender as a Python module: pip install bpy==5.0.1):
    python render/garden_cycles.py --scene out/render_scene --out out/renders \\
        --views front,evening,stream,sand,arm,tree --quality draft|preview|local|final [--device GPU]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import bpy
import bmesh                   # after bpy: the module sets up Blender's own packages
import numpy as np
from mathutils import Euler, Vector, noise

MM = 0.001                     # the model is in mm, the scene in metres (for the sky and the lens)
QUALITY = {                    # moss shoots per mm2, samples, resolution
    "draft": dict(moss=0.6, samples=16, res=(800, 500)),
    "preview": dict(moss=1.8, samples=48, res=(1440, 900)),
    "local": dict(moss=2.6, samples=80, res=(1800, 1125)),     # the best a 4-core CPU does in ~5 min
    "final": dict(moss=4.5, samples=384, res=(2400, 1500)),
}


# ------------------------------------------------------------------ scene data
class Garden:
    def __init__(self, folder: Path):
        self.s = json.loads((folder / "scene.json").read_text())
        t = np.load(folder / "terrain.npz")
        self.ground, self.zone, self.water, self.water_ext = t["ground"], t["zone"], t["water"], t["water_ext"]
        sd = np.load(folder / "sand.npz")
        self.sand_h, self.sand_mask = sd["h"], sd["mask"]
        self.bonsai = dict(np.load(folder / "bonsai.npz"))
        self.W, self.D = self.s["tray"]["width"], self.s["tray"]["depth"]
        self.bed, self.wall = self.s["tray"]["bed"], self.s["tray"]["wall"]

    def V(self, x, y, h=0.0) -> Vector:
        """Garden mm (x right, y away from the viewer, h above the sand) -> scene metres."""
        return Vector(((x - self.W / 2) * MM, (y - self.D / 2) * MM, h * MM))

    def g(self, x, y) -> float:
        """Ground height (mm above the sand), bilinear on the 1 mm grid of cell centres."""
        ny, nx = self.ground.shape
        fx = min(max(x - 0.5, 0), nx - 1.001)
        fy = min(max(y - 0.5, 0), ny - 1.001)
        j, i = int(fx), int(fy)
        u, v = fx - j, fy - i
        G = self.ground
        return float((G[i, j] * (1 - u) + G[i, j + 1] * u) * (1 - v) + (G[i + 1, j] * (1 - u) + G[i + 1, j + 1] * u) * v)


# ------------------------------------------------------------------ mesh helpers
def mesh_object(name, verts, faces, coll=None, smooth=True, uv=None, mat=None):
    """verts (N, 3) metres; faces (M, k) vertex indices, all faces of one size k."""
    me = bpy.data.meshes.new(name)
    verts = np.asarray(verts, np.float32)
    faces = np.asarray(faces, np.int32)
    k = faces.shape[1]
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", verts.ravel())
    me.loops.add(faces.size)
    me.loops.foreach_set("vertex_index", faces.ravel())
    me.polygons.add(len(faces))
    me.polygons.foreach_set("loop_start", np.arange(0, faces.size, k, dtype=np.int32))
    me.update(calc_edges=True)
    if uv is not None:
        layer = me.uv_layers.new(name="UVMap")
        layer.data.foreach_set("uv", np.asarray(uv, np.float32)[faces.ravel()].ravel())
    if smooth:
        me.shade_smooth()
    if mat is not None:
        me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    (coll or bpy.context.scene.collection).objects.link(ob)
    return ob


def grid_object(name, Z, xs, ys, face_mask, gd: Garden, mat=None, uv_scale=None):
    """A heightfield mesh over the grid points (xs, ys in mm; Z in mm above the sand), keeping the
    quads whose cell is in face_mask."""
    ny, nx = Z.shape
    fi, fj = np.nonzero(face_mask[:-1, :-1])
    a = fi * nx + fj
    quads = np.stack([a, a + 1, a + nx + 1, a + nx], 1)
    used, inv = np.unique(quads, return_inverse=True)
    quads = inv.reshape(-1, 4)
    XX, YY = np.meshgrid(xs, ys)
    x, y, z = XX.ravel()[used], YY.ravel()[used], Z.ravel()[used]
    verts = np.stack([(x - gd.W / 2) * MM, (y - gd.D / 2) * MM, z * MM], 1)
    uv = None if uv_scale is None else np.stack([x * uv_scale, y * uv_scale], 1)
    return mesh_object(name, verts, quads, mat=mat, uv=uv)


def icosphere(subdiv=3):
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
         (8, 6, 7), (9, 8, 1)]
    verts = [np.array(p, float) / np.linalg.norm(p) for p in v]
    faces = f
    for _ in range(subdiv):
        cache, nf = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = nf
    return np.array(verts), np.array(faces)


def rock_shape(seed, rough=0.22, subdiv=4, flat=0.0):
    """A unit rock: an icosphere pushed about by fractal noise, flattened underneath."""
    v, f = icosphere(subdiv)
    off = Vector((seed * 7.31, seed * 3.17, seed * 5.53))
    out = np.empty_like(v)
    for i, p in enumerate(v):
        q = Vector(p) * 1.4 + off
        r = 1 + rough * noise.fractal(q, 0.6, 2.2, 5) + 0.06 * noise.noise(q * 4.0)
        out[i] = p * r
    out[:, 2] = np.where(out[:, 2] < 0, out[:, 2] * (1 - flat), out[:, 2])
    return out, f


def tube(points, radii, sides=12, bumps=0.0, seed=0.0, lobes=0.0):
    """A tube along the polyline `points` (N, 3, metres) with per-point radii; UVs run around and along."""
    P = np.asarray(points, float)
    N = len(P)
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    ref = np.array([0, 0, 1.0]) if abs(T[0][2]) < 0.9 else np.array([1.0, 0, 0])
    Nrm = np.cross(T[0], ref)
    Nrm /= np.linalg.norm(Nrm)
    verts, uv = [], []
    length = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    for i in range(N):
        if i:                                              # parallel transport of the frame
            Nrm = Nrm - T[i] * (Nrm @ T[i])
            Nrm /= np.linalg.norm(Nrm)
        B = np.cross(T[i], Nrm)
        for k in range(sides + 1):
            a = 2 * math.pi * k / sides
            t = i / max(N - 1, 1)
            r = radii[i] * (1 + bumps * noise.noise(Vector((P[i][0] * 900 + seed, P[i][1] * 900 + math.cos(a) * 2, P[i][2] * 900 + math.sin(a) * 2)))
                            + lobes * math.sin(2 * a + 4 * t + seed) + 0.6 * lobes * math.sin(3 * a - 3 * t))
            verts.append(P[i] + r * (math.cos(a) * Nrm + math.sin(a) * B))
            uv.append((k / sides, length[i] / (2 * math.pi * max(radii[0], 1e-4))))
    faces = []
    for i in range(N - 1):
        for k in range(sides):
            a = i * (sides + 1) + k
            faces.append((a, a + 1, a + sides + 2, a + sides + 1))
    return np.array(verts), np.array(faces), np.array(uv)


def catmull(points, n=40):
    """Centripetal-ish Catmull-Rom through points (M, 3) -> (n, 3)."""
    P = np.asarray(points, float)
    P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    segs = len(P) - 3
    for s in range(segs):
        p0, p1, p2, p3 = P[s:s + 4]
        for t in np.linspace(0, 1, max(2, n // segs), endpoint=(s == segs - 1)):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return np.array(out)


def new_collection(name, link=True):
    c = bpy.data.collections.new(name)
    if link:
        bpy.context.scene.collection.children.link(c)
    return c


# ------------------------------------------------------------------ shader helpers
class Nodes:
    def __init__(self, tree):
        self.t = tree
        self.n = tree.nodes
        self.l = tree.links

    def new(self, kind, **props):
        node = self.n.new(kind)
        for k, v in props.items():
            setattr(node, k, v)
        return node

    def link(self, a, b):
        self.l.new(a, b)

    @staticmethod
    def sock(sockets, name, kind=None):
        for s in sockets:
            if s.name == name and s.enabled and (kind is None or s.type == kind):
                return s
        raise KeyError(name)

    def set(self, node, **values):
        for k, v in values.items():
            self.sock(node.inputs, k.replace("_", " ")).default_value = v

    def ramp(self, fac, stops):
        r = self.new("ShaderNodeValToRGB")
        els = r.color_ramp.elements
        while len(els) > 1:
            els.remove(els[-1])
        els[0].position, els[0].color = stops[0][0], (*stops[0][1], 1)
        for pos, col in stops[1:]:
            e = els.new(pos)
            e.color = (*col, 1)
        self.link(fac, r.inputs["Fac"])
        return r.outputs["Color"]

    def mix(self, fac, a, b):
        m = self.new("ShaderNodeMix", data_type="RGBA", blend_type="MIX")
        self.link(fac, m.inputs[0]) if not isinstance(fac, float) else setattr(m.inputs[0], "default_value", fac)
        for sock, val in ((m.inputs[6], a), (m.inputs[7], b)):
            if isinstance(val, tuple):
                sock.default_value = (*val, 1)
            else:
                self.link(val, sock)
        return m.outputs[2]

    def math(self, op, a, b=None, clamp=False):
        m = self.new("ShaderNodeMath", operation=op, use_clamp=clamp)
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                m.inputs[i].default_value = v
            else:
                self.link(v, m.inputs[i])
        return m.outputs[0]

    def bump(self, height, strength, distance, normal=None):
        b = self.new("ShaderNodeBump")
        self.set(b, Strength=strength, Distance=distance)
        self.link(height, b.inputs["Height"])
        if normal is not None:
            self.link(normal, b.inputs["Normal"])
        return b.outputs["Normal"]

    def noise(self, vector, scale, detail=4.0, rough=0.55, distortion=0.0):
        t = self.new("ShaderNodeTexNoise")
        self.set(t, Scale=scale, Detail=detail, Roughness=rough, Distortion=distortion)
        if vector is not None:
            self.link(vector, t.inputs["Vector"])
        return t

    def voronoi(self, vector, scale, feature="F1", randomness=1.0):
        t = self.new("ShaderNodeTexVoronoi", feature=feature)
        self.set(t, Scale=scale, Randomness=randomness)
        if vector is not None:
            self.link(vector, t.inputs["Vector"])
        return t

    def mapping(self, vector, scale=(1, 1, 1)):
        m = self.new("ShaderNodeMapping")
        m.inputs["Scale"].default_value = scale
        self.link(vector, m.inputs["Vector"])
        return m.outputs["Vector"]

    def coords(self, which="Object"):
        return self.new("ShaderNodeTexCoord").outputs[which]


def material(name, build):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    N = Nodes(nt)
    out = N.new("ShaderNodeOutputMaterial")
    bsdf = N.new("ShaderNodeBsdfPrincipled")
    N.link(bsdf.outputs["BSDF"], out.inputs["Surface"])
    build(N, bsdf, out)
    return m


def wood_build(dark, mid, light, rings=260.0, coat=0.5, rough=0.42, figure_scale=6.0, line=0.55, distortion=6.0):
    """Face grain along UV u: long latewood lines (stretched noise feeding wave bands), pores, figure."""
    def build(N, bsdf, out):
        uv = N.coords("UV")
        stretched = N.mapping(uv, (1.0, 38.0, 1.0))                 # features long along the board
        wave = N.new("ShaderNodeTexWave", wave_type="BANDS", bands_direction="Y", wave_profile="SIN")
        N.set(wave, Scale=rings / 20.0, Distortion=distortion, Detail=3.0, Detail_Scale=1.2)
        N.link(uv, wave.inputs["Vector"])
        late = N.math("POWER", wave.outputs["Fac"], 3.0)
        fig = N.noise(stretched, figure_scale, 6.0, 0.62)
        pores = N.noise(N.mapping(uv, (60.0, 900.0, 60.0)), 1.0, 1.0, 0.5)
        pore = N.math("GREATER_THAN", pores.outputs["Fac"], 0.66)
        base = N.ramp(fig.outputs["Fac"], [(0.3, light), (0.55, mid), (0.8, dark)])
        col = N.mix(N.math("MULTIPLY", late, line), base, dark)
        col = N.mix(N.math("MULTIPLY", pore, 0.35), col, dark)
        N.link(col, bsdf.inputs["Base Color"])
        N.set(bsdf, Roughness=rough, Coat_Weight=coat, Coat_Roughness=0.16)
        h = N.math("ADD", N.math("MULTIPLY", late, -0.6), N.math("MULTIPLY", pore, -1.0))
        N.link(N.bump(h, 0.08, 0.0004), bsdf.inputs["Normal"])
    return build


def make_materials(q):
    M = {}
    M["walnut"] = material("walnut", wood_build((0.02, 0.0095, 0.005), (0.052, 0.027, 0.014), (0.1, 0.054, 0.03), coat=0.22, rough=0.5, distortion=2.5))
    M["oak"] = material("oak", wood_build((0.25, 0.17, 0.1), (0.36, 0.26, 0.16), (0.44, 0.33, 0.21), rings=520, coat=0.0, rough=0.62, line=0.16))
    M["cedar"] = material("cedar", wood_build((0.12, 0.05, 0.025), (0.24, 0.11, 0.055), (0.36, 0.18, 0.09), rings=180, coat=0.0, rough=0.7))

    def sand(N, bsdf, out):
        p = N.coords("Object")
        grains = N.voronoi(p, 3200.0)                                 # ~0.3 mm quartz grains
        speck = N.new("ShaderNodeSeparateColor")
        N.link(grains.outputs["Color"], speck.inputs["Color"])
        tint = N.math("MULTIPLY", N.sock(speck.outputs, "Red"), 0.12)
        col = N.mix(tint, (0.60, 0.575, 0.51), (0.45, 0.42, 0.36))
        fines = N.noise(p, 900.0, 3.0, 0.6)
        col = N.mix(N.math("MULTIPLY", fines.outputs["Fac"], 0.2), col, (0.66, 0.64, 0.58))
        col = N.mix(N.math("GREATER_THAN", N.sock(speck.outputs, "Blue"), 0.988), col, (0.1, 0.095, 0.09))
        N.link(col, bsdf.inputs["Base Color"])
        glint = N.math("GREATER_THAN", N.sock(speck.outputs, "Green"), 0.965)
        N.link(N.math("SUBTRACT", 0.9, N.math("MULTIPLY", glint, 0.75)), bsdf.inputs["Roughness"])
        N.link(N.math("MULTIPLY", glint, 0.9), bsdf.inputs["Specular IOR Level"])
        h = N.math("SUBTRACT", 1.0, N.math("POWER", grains.outputs["Distance"], 0.6))
        N.link(N.bump(h, 0.35, 0.00012), bsdf.inputs["Normal"])
    M["sand"] = material("sand", sand)

    def moss_leaf(tip, base, variety, patch):
        def build(N, bsdf, out):
            info = N.new("ShaderNodeObjectInfo")
            gen = N.coords("Generated")
            sep = N.new("ShaderNodeSeparateXYZ")
            N.link(gen, sep.inputs["Vector"])
            along = sep.outputs["Z"]
            col = N.mix(along, base, tip)
            col = N.mix(N.math("MULTIPLY", info.outputs["Random"], variety), col, (base[0] * 0.55, base[1] * 0.7, base[2] * 0.5))
            pn = N.noise(info.outputs["Location"], 38.0, 3.0, 0.5)
            col = N.mix(N.math("MULTIPLY", N.math("GREATER_THAN", pn.outputs["Fac"], 0.52), 0.55), col, patch)
            N.link(col, bsdf.inputs["Base Color"])
            N.set(bsdf, Roughness=0.62, Subsurface_Weight=0.0, Transmission_Weight=0.0)
            trans = N.new("ShaderNodeBsdfTranslucent")
            N.link(col, trans.inputs["Color"])
            mix = N.new("ShaderNodeMixShader")
            mix.inputs[0].default_value = 0.28
            N.link(bsdf.outputs["BSDF"], mix.inputs[1])
            N.link(trans.outputs["BSDF"], mix.inputs[2])
            N.link(mix.outputs[0], out.inputs["Surface"])
        return build
    M["moss_living"] = material("moss_living", moss_leaf((0.17, 0.3, 0.045), (0.03, 0.075, 0.012), 0.6, (0.2, 0.26, 0.04)))
    M["moss_preserved"] = material("moss_preserved", moss_leaf((0.2, 0.36, 0.07), (0.05, 0.13, 0.03), 0.45, (0.09, 0.22, 0.06)))

    def ground(N, bsdf, out):
        p = N.coords("Object")
        n = N.noise(p, 250.0, 5.0, 0.6)
        col = N.ramp(n.outputs["Fac"], [(0.35, (0.012, 0.018, 0.006)), (0.65, (0.04, 0.055, 0.018))])
        N.link(col, bsdf.inputs["Base Color"])
        N.set(bsdf, Roughness=0.95)
        N.link(N.bump(n.outputs["Fac"], 0.5, 0.0008), bsdf.inputs["Normal"])
    M["moss_ground"] = material("moss_ground", ground)

    def bed(N, bsdf, out):
        p = N.coords("Object")
        n = N.noise(p, 400.0, 5.0, 0.6)
        g = N.voronoi(p, 700.0)
        col = N.ramp(n.outputs["Fac"], [(0.3, (0.07, 0.06, 0.042)), (0.7, (0.17, 0.15, 0.11))])
        col = N.mix(N.math("MULTIPLY", g.outputs["Distance"], 0.6), col, (0.04, 0.045, 0.03))
        N.link(col, bsdf.inputs["Base Color"])
        N.set(bsdf, Roughness=0.35)
        N.link(N.bump(g.outputs["Distance"], 0.4, 0.0008), bsdf.inputs["Normal"])
    M["bed"] = material("bed", bed)

    def water(N, bsdf, out):
        p = N.coords("Object")
        n1 = N.noise(N.mapping(p, (1.0, 1.0, 1.0)), 180.0, 3.0, 0.5)
        n2 = N.noise(p, 900.0, 2.0, 0.5)
        h = N.math("ADD", n1.outputs["Fac"], N.math("MULTIPLY", n2.outputs["Fac"], 0.25))
        N.set(bsdf, Base_Color=(0.9, 0.97, 0.95, 1), Roughness=0.015, IOR=1.333, Transmission_Weight=1.0)
        N.link(N.bump(h, 0.1, 0.0006), bsdf.inputs["Normal"])
    M["water"] = material("water", water)

    def sheet(N, bsdf, out):
        uv = N.coords("UV")
        streak = N.noise(N.mapping(uv, (16.0, 1.0, 1.0)), 6.0, 5.0, 0.6)
        sr = N.new("ShaderNodeMapRange")
        N.set(sr, From_Min=0.55, From_Max=0.75, To_Min=0.0, To_Max=0.65)
        N.link(streak.outputs["Fac"], sr.inputs["Value"])
        white = sr.outputs["Result"]
        N.set(bsdf, Base_Color=(0.95, 0.98, 0.98, 1), Roughness=0.05, IOR=1.333, Transmission_Weight=1.0)
        foam = N.new("ShaderNodeBsdfPrincipled")
        N.set(foam, Base_Color=(0.85, 0.88, 0.88, 1), Roughness=0.4, Transmission_Weight=0.4)
        mix = N.new("ShaderNodeMixShader")
        N.link(white, mix.inputs[0])
        N.link(bsdf.outputs["BSDF"], mix.inputs[1])
        N.link(foam.outputs["BSDF"], mix.inputs[2])
        N.link(mix.outputs[0], out.inputs["Surface"])
    M["sheet"] = material("cascade", sheet)

    def foam(N, bsdf, out):
        p = N.coords("Object")
        n = N.noise(p, 900.0, 6.0, 0.7)
        v = N.voronoi(p, 2400.0)
        mask = N.math("MULTIPLY", N.math("GREATER_THAN", n.outputs["Fac"], 0.56), N.math("GREATER_THAN", v.outputs["Distance"], 0.25))
        N.set(bsdf, Base_Color=(0.88, 0.9, 0.9, 1), Roughness=0.5)
        N.link(N.math("MULTIPLY", mask, 0.8), bsdf.inputs["Alpha"])
    M["foam"] = material("foam", foam)

    def rockmat(base, spots, moss_on_top):
        def build(N, bsdf, out):
            p = N.coords("Object")
            n1 = N.noise(p, 120.0, 6.0, 0.6)
            n2 = N.noise(p, 900.0, 4.0, 0.55)
            vor = N.voronoi(p, 2600.0)
            sep = N.new("ShaderNodeSeparateColor")
            N.link(vor.outputs["Color"], sep.inputs["Color"])
            info = N.new("ShaderNodeObjectInfo")
            col = N.ramp(n1.outputs["Fac"], [(0.3, tuple(c * 0.6 for c in base)), (0.7, base)])
            col = N.mix(N.math("MULTIPLY", N.math("GREATER_THAN", N.sock(sep.outputs, "Red"), 0.8), 0.35), col, spots)
            tone = N.new("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
            tone.inputs[0].default_value = 1.0
            N.link(col, tone.inputs[6])
            N.link(N.mix(N.math("MULTIPLY", info.outputs["Random"], 0.6), info.outputs["Color"], (0.5, 0.5, 0.5)), tone.inputs[7])
            col = tone.outputs[2]
            warp = N.noise(p, 400.0, 4.0, 0.6, distortion=0.8)
            lichen = N.voronoi(warp.outputs["Color"], 6.0)
            lmask = N.math("MULTIPLY", N.math("LESS_THAN", lichen.outputs["Distance"], 0.2), N.math("GREATER_THAN", N.noise(p, 60.0).outputs["Fac"], 0.58))
            col = N.mix(N.math("MULTIPLY", lmask, 0.45), col, (0.26, 0.28, 0.22))
            rough = 0.82
            if moss_on_top:
                geo = N.new("ShaderNodeNewGeometry")
                nz = N.new("ShaderNodeSeparateXYZ")
                N.link(geo.outputs["Normal"], nz.inputs["Vector"])
                up = N.math("SUBTRACT", nz.outputs["Z"], 0.45)
                cover = N.math("GREATER_THAN", N.math("ADD", up, N.math("MULTIPLY", n1.outputs["Fac"], 0.5)), 0.52)
                mc = N.ramp(n2.outputs["Fac"], [(0.35, (0.03, 0.07, 0.012)), (0.7, (0.12, 0.2, 0.03))])
                col = N.mix(cover, col, mc)
            N.link(col, bsdf.inputs["Base Color"])
            N.set(bsdf, Roughness=rough)
            h = N.math("ADD", n1.outputs["Fac"], N.math("MULTIPLY", n2.outputs["Fac"], 0.5))
            N.link(N.bump(h, 0.45, 0.002), bsdf.inputs["Normal"])
        return build
    M["stone"] = material("stone", rockmat((0.09, 0.085, 0.078), (0.05, 0.048, 0.045), False))
    M["rock"] = material("rock", rockmat((0.2, 0.19, 0.17), (0.12, 0.115, 0.1), True))
    M["pebble"] = material("pebble", rockmat((0.3, 0.28, 0.25), (0.18, 0.17, 0.15), False))

    def granite(N, bsdf, out):
        p = N.coords("Object")
        vor = N.voronoi(p, 1700.0)
        sep = N.new("ShaderNodeSeparateColor")
        N.link(vor.outputs["Color"], sep.inputs["Color"])
        base = N.ramp(N.noise(p, 220.0, 4.0, 0.6).outputs["Fac"], [(0.35, (0.28, 0.27, 0.25)), (0.7, (0.4, 0.39, 0.36))])
        col = N.mix(N.math("GREATER_THAN", N.sock(sep.outputs, "Red"), 0.86), base, (0.03, 0.03, 0.03))
        col = N.mix(N.math("GREATER_THAN", N.sock(sep.outputs, "Green"), 0.88), col, (0.58, 0.46, 0.42))
        col = N.mix(N.math("GREATER_THAN", N.sock(sep.outputs, "Blue"), 0.8), col, (0.7, 0.69, 0.66))
        N.link(col, bsdf.inputs["Base Color"])
        N.set(bsdf, Roughness=0.78)
        h = N.math("ADD", vor.outputs["Distance"], N.math("MULTIPLY", N.noise(p, 600.0, 3.0).outputs["Fac"], 0.6))
        N.link(N.bump(h, 0.25, 0.0005), bsdf.inputs["Normal"])
    M["granite"] = material("granite", granite)

    def bark(N, bsdf, out):
        uv = N.coords("UV")
        warp = N.noise(N.mapping(uv, (2.0, 0.5, 1.0)), 3.0, 3.0, 0.5, distortion=0.6)
        plates = N.new("ShaderNodeTexVoronoi", feature="DISTANCE_TO_EDGE")
        N.set(plates, Scale=4.0, Randomness=0.9)
        vm = N.new("ShaderNodeMix", data_type="VECTOR")
        vm.inputs[0].default_value = 0.25
        N.link(N.mapping(uv, (14.0, 4.5, 1.0)), N.sock(vm.inputs, "A", "VECTOR"))
        N.link(warp.outputs["Color"], N.sock(vm.inputs, "B", "VECTOR"))
        N.link(N.sock(vm.outputs, "Result", "VECTOR"), plates.inputs["Vector"])
        edge = plates.outputs["Distance"]                     # 0 in a fissure, rising onto a plate
        plate = N.math("POWER", N.math("MULTIPLY", edge, 6.0, clamp=True), 0.5)
        fine = N.noise(N.mapping(uv, (8.0, 2.0, 1.0)), 12.0, 6.0, 0.62)
        h = N.math("ADD", plate, N.math("MULTIPLY", fine.outputs["Fac"], 0.35))
        col = N.ramp(plate, [(0.05, (0.007, 0.005, 0.004)), (0.45, (0.04, 0.023, 0.014)), (0.9, (0.07, 0.042, 0.027))])
        grey = N.math("GREATER_THAN", fine.outputs["Fac"], 0.55)
        col = N.mix(N.math("MULTIPLY", N.math("MULTIPLY", grey, plate), 0.55), col, (0.11, 0.1, 0.09))
        N.link(col, bsdf.inputs["Base Color"])
        N.set(bsdf, Roughness=0.9)
        N.link(N.bump(h, 1.0, 0.004), bsdf.inputs["Normal"])
    M["bark"] = material("bark", bark)

    def needles(N, bsdf, out):
        info = N.new("ShaderNodeObjectInfo")
        gen = N.new("ShaderNodeSeparateXYZ")
        N.link(N.coords("Generated"), gen.inputs["Vector"])
        col = N.ramp(info.outputs["Random"], [(0.0, (0.012, 0.035, 0.01)), (0.6, (0.025, 0.06, 0.016)), (1.0, (0.05, 0.1, 0.025))])
        col = N.mix(N.math("POWER", gen.outputs["Z"], 3.0), col, (0.1, 0.16, 0.04))
        N.link(col, bsdf.inputs["Base Color"])
        N.set(bsdf, Roughness=0.5)
        trans = N.new("ShaderNodeBsdfTranslucent")
        N.link(col, trans.inputs["Color"])
        mix = N.new("ShaderNodeMixShader")
        mix.inputs[0].default_value = 0.2
        N.link(bsdf.outputs["BSDF"], mix.inputs[1])
        N.link(trans.outputs["BSDF"], mix.inputs[2])
        N.link(mix.outputs[0], out.inputs["Surface"])
    M["needles"] = material("needles", needles)

    M["brass"] = material("brass", lambda N, b, o: N.set(b, Base_Color=(0.78, 0.56, 0.28, 1), Metallic=1.0, Roughness=0.3))
    M["steel"] = material("steel", lambda N, b, o: N.set(b, Base_Color=(0.66, 0.66, 0.68, 1), Metallic=1.0, Roughness=0.22))
    M["kerb"] = material("kerb", lambda N, b, o: (N.set(b, Base_Color=(0.03, 0.03, 0.028, 1), Roughness=0.5),
                                                  N.link(N.bump(N.noise(N.coords("Object"), 500.0).outputs["Fac"], 0.3, 0.0004), b.inputs["Normal"])))
    M["plaster"] = material("plaster", lambda N, b, o: N.set(b, Base_Color=(0.62, 0.6, 0.56, 1), Roughness=0.95))

    return M


def paper_material(name, rgb):
    """Washi in a firebox window: matt off-white, and when the lantern is on, the light its LED
    sends out through the paper (strength set by lighting(); sampled from the outside face only)."""
    def build(N, bsdf, out):
        N.set(bsdf, Base_Color=(0.8, 0.76, 0.66, 1), Roughness=0.92, Emission_Color=(*rgb, 1.0), Emission_Strength=0.0)
    m = material(name, build)
    m.cycles.emission_sampling = "FRONT"
    return m


# ------------------------------------------------------------------ the garden
def build_tray(gd: Garden, M):
    W, D, S = gd.W, gd.D, gd.bed
    T, base_t, plinth, inset = 22.0, 12.0, 50.0, 10.0
    top, lo = gd.wall - S, -S - base_t
    r = 5.0
    prof = [(0.0, lo), (0.0, top - r)]
    prof += [(r - r * math.cos(a), top - r + r * math.sin(a)) for a in np.linspace(0, math.pi / 2, 9)[1:]]
    prof += [(T - r + r * math.sin(a), top - r + r * math.cos(a)) for a in np.linspace(0, math.pi / 2, 9)]
    prof += [(T, lo)]
    arc = np.concatenate([[0], np.cumsum([math.dist(prof[k], prof[k - 1]) for k in range(1, len(prof))])])
    for name, p0, p1, out in (("front", (0, 0), (W, 0), (0, -1)), ("right", (W, 0), (W, D), (1, 0)),
                              ("back", (W, D), (0, D), (0, 1)), ("left", (0, D), (0, 0), (-1, 0))):
        u = np.subtract(p1, p0) / math.dist(p0, p1)
        L = math.dist(p0, p1)
        verts, uv = [], []
        for end in (0, 1):
            for k, (a, h) in enumerate(prof):
                s = L + a if end else -a
                x, y = p0[0] + u[0] * s + out[0] * a, p0[1] + u[1] * s + out[1] * a
                verts.append(gd.V(x, y, h))
                uv.append((s / 280.0, arc[k] / 280.0 + 0.37 * (p0[0] + p0[1]) / 280.0))
        m = len(prof)
        faces = [(k, k + 1, m + k + 1, m + k) for k in range(m - 1)]
        ob = mesh_object("frame_" + name, verts, faces, uv=uv, mat=M["walnut"], smooth=True)
        n0 = ob.data.polygons[m - 2].normal                         # the outer face must face out
        if n0.x * out[0] + n0.y * out[1] < 0:
            ob.data.flip_normals()
    # base plate inside the frame, plinth set back, table, a wall behind
    def box(name, cx, cy, cz, sx, sy, sz, mat, uvs=0.3):
        v = np.array([[x, y, z] for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)], float) * 0.5 * np.array([sx, sy, sz])
        v += np.array([cx, cy, cz])
        f = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        uv = v[:, :2] / uvs
        ob = mesh_object(name, v, f, uv=uv, mat=mat, smooth=False)
        return ob
    c = gd.V(W / 2, D / 2)
    box("base", c.x, c.y, (lo + base_t / 2) * MM, W * MM, D * MM, base_t * MM, M["walnut"])
    pw, pd = (W + 2 * (T - inset)) * MM, (D + 2 * (T - inset)) * MM
    box("plinth", c.x, c.y, (lo - plinth / 2) * MM, pw, pd, plinth * MM, M["walnut"])
    table_z = (lo - plinth) * MM
    box("table", c.x, c.y + 0.05, table_z - 0.015, 1.9, 1.3, 0.03, M["oak"], uvs=0.32)
    wall = box("wall", c.x, c.y + 1.15, table_z + 0.8, 4.0, 0.05, 2.2, M["plaster"])
    return table_z, wall


def build_ground(gd: Garden, M):
    ny, nx = gd.ground.shape
    xs, ys = np.arange(nx) + 0.5, np.arange(ny) + 0.5
    Z = gd.zone
    corners = [Z[:-1, :-1], Z[:-1, 1:], Z[1:, :-1], Z[1:, 1:]]
    anyof = lambda codes: np.logical_or.reduce([np.isin(z, codes) for z in corners])
    allof = lambda codes: np.logical_and.reduce([np.isin(z, codes) for z in corners])
    sandy = anyof((0, 6))                                  # the sand mesh and the kerb cover these
    living = anyof((2, 4, 5)) & ~sandy
    preserved = anyof((3,)) & ~living & ~sandy
    kerb = anyof((7,)) & ~living & ~preserved & ~sandy
    bed = allof((1,))
    pad = lambda m: np.pad(m, ((0, 1), (0, 1)))
    obs = {}
    G = np.where(Z == 7, np.minimum(gd.ground, 2.0), gd.ground)      # tucked under the swept kerb
    for name, mask, mat in (("moss_living", living, M["moss_ground"]), ("moss_preserved", preserved, M["moss_ground"]),
                            ("kerb", kerb, M["moss_ground"]), ("bed", bed, M["bed"])):
        obs[name] = grid_object("ground_" + name, G, xs, ys, pad(mask), gd, mat=mat)
    # the water: surface carried a little under the banks so it meets them without a gap. It casts
    # no shadow: shallow clear water lets the sun through to its bed (with refractive caustics off,
    # Cycles would otherwise leave the bed in shadow)
    wet = ~np.isnan(gd.water_ext)
    Zw = np.where(wet, gd.water_ext, -30.0)
    obs["water"] = grid_object("water", Zw, xs, ys, wet & np.roll(wet, -1, 0) & np.roll(wet, -1, 1), gd, mat=M["water"])
    obs["water"].visible_shadow = False
    return obs


def build_sand(gd: Garden, M):
    s = gd.s["sand"]
    xs = s["x0"] + (np.arange(s["nx"]) + 0.5) * s["dx"]
    ys = s["y0"] + (np.arange(s["ny"]) + 0.5) * s["dx"]
    mask = gd.sand_mask
    faces = mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]
    full = np.zeros_like(mask)
    full[:-1, :-1] = faces
    return grid_object("sand", gd.sand_h, xs, ys, full, gd, mat=M["sand"])


def build_kerb(gd: Garden, M):
    loop = np.array(gd.s["sand"]["outline"], float)
    n = len(loop)
    area = sum(loop[i][0] * loop[(i + 1) % n][1] - loop[(i + 1) % n][0] * loop[i][1] for i in range(n))
    turn = 1 if area > 0 else -1
    prof = [(-1.2, -1.5), (-1.2, 2.4)]                       # the kerb band is 5 mm; this covers 7
    prof += [(0.0 - 1.2 * math.cos(a), 2.4 + 1.2 * math.sin(a)) for a in np.linspace(0, math.pi / 2, 5)[1:]]
    prof += [(5.0 + 1.2 * math.sin(a), 2.4 + 1.2 * math.cos(a)) for a in np.linspace(0, math.pi / 2, 5)]
    prof += [(6.6, 1.2), (7.2, -0.5)]
    verts = []
    dense = []
    for i in range(n):                                     # resample the outline every ~2 mm
        a, b = loop[i], loop[(i + 1) % n]
        k = max(1, int(math.dist(a, b) / 2))
        dense += [a + (b - a) * t for t in np.arange(k) / k]
    dense = np.array(dense)
    m = len(dense)
    for i in range(m + 1):
        p, a, b = dense[i % m], dense[(i - 1) % m], dense[(i + 1) % m]
        t = b - a
        t /= np.linalg.norm(t)
        o = np.array([turn * t[1], -turn * t[0]])
        for (d, h) in prof:
            q = p + o * d
            verts.append(gd.V(q[0], q[1], h))
    k = len(prof)
    faces = [(i * k + j, i * k + j + 1, (i + 1) * k + j + 1, (i + 1) * k + j) for i in range(m) for j in range(k - 1)]
    ob = mesh_object("kerb_edge", verts, faces, mat=M["kerb"])
    if ob.data.polygons[k // 2].normal.z < 0:
        ob.data.flip_normals()
    return ob


def rock_object(name, gd: Garden, outline, top, seed, mat, rough=0.2, sink=3.0, coll=None):
    pts = np.array(outline, float)
    c = pts.mean(0)
    best, ax = 0, 0
    for a in np.linspace(0, math.pi, 72, endpoint=False):
        d = np.array([math.cos(a), math.sin(a)])
        e = np.abs((pts - c) @ d).max()
        if e > best:
            best, ax = e, a
    d1 = np.array([math.cos(ax), math.sin(ax)])
    d2 = np.array([-d1[1], d1[0]])
    r1, r2 = np.abs((pts - c) @ d1).max(), np.abs((pts - c) @ d2).max()
    low = min(gd.g(*p) for p in pts)
    v, f = rock_shape(seed, rough=rough, subdiv=5, flat=0.6)
    up = np.maximum(v[:, 2], -0.15)
    hgt = low - sink + np.power(np.maximum(up, 0), 0.8) * (top - low + sink) + np.minimum(v[:, 2], 0) * 6
    xy = c + np.outer(v[:, 0] * r1, d1) + np.outer(v[:, 1] * r2, d2)
    verts = [gd.V(x, y, h) for (x, y), h in zip(xy, hgt)]
    return mesh_object(name, verts, f, mat=mat, coll=coll)


def build_stones_and_rocks(gd: Garden, M):
    for i, s in enumerate(gd.s["stones"]):
        rock_object("stone_" + s["name"], gd, s["outline"], s["top"], 3 + i * 7, M["stone"], rough=0.16, sink=4.0)
    for i, r in enumerate(gd.s["rocks"]):
        rock_object("rock_" + r["name"], gd, r["outline"], r["top"], 11 + i * 5, M["rock"], rough=0.24)
    # pebbles and edge stones: a few shapes, many linked copies
    coll = new_collection("pebbles")
    shapes = [rock_shape(40 + k, rough=0.18, subdiv=3, flat=0.3) for k in range(4)]
    meshes = []
    for k, (v, f) in enumerate(shapes):
        ob = mesh_object(f"pebble_shape_{k}", v, f, mat=M["pebble"], coll=coll)
        meshes.append(ob.data)
        coll.objects.unlink(ob)
    rng = np.random.default_rng(3)
    for group in ("edge_stones", "bed_pebbles"):
        for n, (x, y, z, sx, sy, sz, rot, tone) in enumerate(gd.s[group]):
            ob = bpy.data.objects.new(f"{group}_{n}", meshes[n % len(meshes)])
            ob.location = gd.V(x, y, z)
            ob.rotation_euler = Euler((rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), rot))
            ob.scale = (sx / 2 * MM, sy / 2 * MM, sz * MM)
            ob.color = (tone, tone, tone, 1)
            coll.objects.link(ob)


def build_cascades(gd: Garden, M):
    for i, cs in enumerate(gd.s["cascades"]):
        d = np.array(cs["dir"])
        perp = np.array([-d[1], d[0]])
        width = min(34.0, 10 + cs["height"])
        out = 4 + cs["height"] * 0.25
        nu, nv = 10, 14
        verts, uv = [], []
        for a in range(nv + 1):
            s = a / nv
            o = out * math.sin(s * math.pi / 2)
            h = cs["top"] - (cs["top"] - cs["bottom"]) * s * s * (0.35 + 0.65 * s)
            for b in range(nu + 1):
                t = b / nu - 0.5
                p = np.array(cs["lip"]) + d * (o - 3) + perp * t * width * (1 + 0.25 * s)
                verts.append(gd.V(p[0], p[1], h + 0.3 + 0.6 * math.sin(t * 9 + i)))
                uv.append((b / nu, s * cs["height"] / 30))
        faces = [(a * (nu + 1) + b, a * (nu + 1) + b + 1, (a + 1) * (nu + 1) + b + 1, (a + 1) * (nu + 1) + b)
                 for a in range(nv) for b in range(nu)]
        ob = mesh_object(f"cascade_{i}", verts, faces, uv=uv, mat=M["sheet"])
        sol = ob.modifiers.new("thick", "SOLIDIFY")
        sol.thickness = 0.8 * MM
        ob.visible_shadow = False
        land = np.array(cs["lip"]) + d * (out + 5)
        ring = [(land[0] + width * 0.5 * math.cos(a) * (1 + 0.15 * math.sin(3 * a + i)),
                 land[1] + width * 0.4 * math.sin(a) * (1 + 0.15 * math.cos(2 * a)))
                for a in np.linspace(0, 2 * math.pi, 40, endpoint=False)]
        verts = [gd.V(land[0], land[1], cs["bottom"] + 0.35)] + [gd.V(x, y, cs["bottom"] + 0.25) for x, y in ring]
        faces = [(0, k + 1, (k + 1) % 40 + 1) for k in range(40)]
        foam = mesh_object(f"foam_{i}", verts, faces, mat=M["foam"])
        foam.visible_shadow = False


def build_lanterns(gd: Garden, M, lights):
    for lan in gd.s["lanterns"]:
        x, y = lan["xy"]
        r = lan["radius"]
        base = lan["foot"] + 1.0
        if base - lan["low"] > 5:                             # a base stone where the footing slopes
            rock_object(f"basestone_{lan['name']}", gd, [(x + r * 1.2 * math.cos(a), y + r * 1.2 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 9)[:-1]],
                        base, 91 + len(lights), M["granite"], rough=0.1, sink=4.0)
        H, LED = lan["top"] - base, lan["light"] - base
        fb0, fb1 = LED - 0.12 * H, LED + 0.1 * H
        parts = [("foot", r * 0.95, r * 0.9, 0, 0.1 * H, 6), ("post", r * 0.34, r * 0.3, 0.1 * H, fb0 - 0.07 * H, 16),
                 ("plate", r * 0.72, r * 0.62, fb0 - 0.07 * H, fb0, 6), ("firebox", r * 0.52, r * 0.52, fb0, fb1, 6),
                 ("eave", r * 1.12, r * 1.05, fb1, fb1 + 0.025 * H, 6)]
        for name, r0, r1, z0, z1, sides in parts:
            prism(f"lantern_{lan['name']}_{name}", gd, x, y, base + z0, base + z1, r0, r1, sides, M["granite"])
        prism(f"lantern_{lan['name']}_roof", gd, x, y, base + fb1 + 0.025 * H, base + fb1 + 0.17 * H, r * 1.08, r * 0.12, 6, M["granite"])
        v, f = icosphere(2)
        jewel = mesh_object(f"lantern_{lan['name']}_jewel", [gd.V(x + p[0] * r * 0.2, y + p[1] * r * 0.2, base + fb1 + 0.19 * H + p[2] * r * 0.26) for p in v], f, mat=M["granite"])
        # The LED sits inside the solid firebox, so its light reaches the garden only through the
        # paper windows: they are the light source, giving out the lantern's lumens between them
        paper = paper_material(f"paper_{lan['name']}", lan["rgb"])
        area = panes(f"lantern_{lan['name']}_windows", gd, x, y, base + fb0 + 0.025 * H, base + fb1 - 0.025 * H, r * 0.535, 0.7, paper)
        lights.append((paper, lan["lumens"] * LUX_TO_W / (math.pi * area)))     # Lambertian: radiance = flux / (pi A)


def panes(name, gd: Garden, x, y, z0, z1, R, frac, mat):
    """A paper pane over the middle `frac` of each face of the hexagonal firebox, which is left as
    stone posts at the corners. Faces outward, a hair proud of the stone. Returns the pane area, m2."""
    ang = np.linspace(0, 2 * math.pi, 6, endpoint=False) + math.pi / 6    # the hexagon prism() makes
    c = [(x + R * math.cos(a), y + R * math.sin(a)) for a in ang]
    verts, faces = [], []
    for k in range(6):
        (ax, ay), (bx, by) = c[k], c[(k + 1) % 6]
        faces.append(tuple(range(len(verts), len(verts) + 4)))
        for t, z in ((0.5 - frac / 2, z0), (0.5 + frac / 2, z0), (0.5 + frac / 2, z1), (0.5 - frac / 2, z1)):
            verts.append(gd.V(ax + (bx - ax) * t, ay + (by - ay) * t, z))
    mesh_object(name, verts, faces, mat=mat, smooth=False)
    return 6 * frac * R * (z1 - z0) * MM * MM


def prism(name, gd: Garden, x, y, z0, z1, r0, r1, sides, mat, caps=True):
    ang = np.linspace(0, 2 * math.pi, sides, endpoint=False) + (math.pi / sides if sides == 6 else 0)
    verts = [gd.V(x + r0 * math.cos(a), y + r0 * math.sin(a), z0) for a in ang] + \
            [gd.V(x + r1 * math.cos(a), y + r1 * math.sin(a), z1) for a in ang]
    faces = [(k, (k + 1) % sides, sides + (k + 1) % sides, sides + k) for k in range(sides)]
    ob = mesh_object(name, verts, faces, mat=mat, smooth=sides > 8)
    if caps:                                               # close both ends with n-gons
        bm = bmesh.new()
        bm.from_mesh(ob.data)
        bm.verts.ensure_lookup_table()
        bm.faces.new([bm.verts[k] for k in reversed(range(sides))])
        bm.faces.new([bm.verts[sides + k] for k in range(sides)])
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(ob.data)
        bm.free()
    bev = ob.modifiers.new("soft", "BEVEL")
    bev.width, bev.segments, bev.limit_method = 0.5 * MM, 2, "ANGLE"
    return ob


def build_bridge(gd: Garden, M):
    for f in gd.s["bridge"]:
        p = np.array(f["outline"], float)
        e1, e2 = p[1] - p[0], p[2] - p[1]
        along, across = (e1, e2) if np.linalg.norm(e1) > np.linalg.norm(e2) else (e2, e1)
        c = p.mean(0)
        L, Wd = np.linalg.norm(along), np.linalg.norm(across)
        u, w = along / L, across / Wd
        h0, h1 = gd.g(*(c - u * L / 2)), gd.g(*(c + u * L / 2))
        rise = max(10.0, f["height"] - max(h0, h1) + 4)
        deck = lambda s: h0 + (h1 - h0) * s + rise * math.sin(math.pi * s)
        planks = 15
        for k in range(planks):
            s0, s1 = k / planks + 0.004, (k + 1) / planks - 0.004
            corners = []
            for s in (s0, s1):
                for side in (-1, 1):
                    q = c + u * (s - 0.5) * L + w * side * Wd / 2
                    corners.append((q, deck(s)))
            verts = [gd.V(q[0], q[1], z + dz) for dz in (-2.8, 0.2) for (q, z) in corners]
            faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
            ob = mesh_object(f"plank_{k}", verts, faces, mat=M["cedar"], smooth=False,
                             uv=[(0.02 * k + i * 0.05, 0.01 * i) for i in range(8)])
            ob.modifiers.new("soft", "BEVEL").width = 0.3 * MM
        for side in (-1, 1):
            rail = [gd.V(*(c + u * (s - 0.5) * L + w * side * Wd / 2), deck(s) + 12) for s in np.linspace(0, 1, 24)]
            v, fc, uv = tube(np.array(rail), [1.3 * MM] * len(rail), sides=8)
            mesh_object(f"rail_{side}", v, fc, uv=uv, mat=M["cedar"])
            for s in (0.06, 0.35, 0.65, 0.94):
                q = c + u * (s - 0.5) * L + w * side * Wd / 2
                v, fc, uv = tube(np.array([gd.V(q[0], q[1], deck(s) - 2), gd.V(q[0], q[1], deck(s) + 12.5)]), [1.5 * MM] * 2, sides=8)
                mesh_object(f"post_{side}_{s}", v, fc, uv=uv, mat=M["cedar"])


def build_tree(gd: Garden, M):
    """The bonsai grown by karesansui.bonsai: bark-textured wood, one mesh for trunk, roots, branches
    and twigs, and needle tufts instanced on the twigs."""
    B = gd.bonsai
    pts, rad, lens = B["branch_pts"], B["branch_r"], B["branch_len"]
    verts, faces, uvs, off, k = [], [], [], 0, 0
    for n in lens:
        P, R = pts[k:k + n], rad[k:k + n]
        k += n
        if n < 2:
            continue
        sides = 24 if R[0] > 8 else 12 if R[0] > 2 else 6
        v, f, uv = tube(np.array([gd.V(*p) for p in P]), R * MM, sides=sides,
                        bumps=0.12 if R[0] > 8 else 0.06, seed=float(k), lobes=0.09 if R[0] > 8 else 0.0)
        verts.append(v)
        faces.append(f + off)
        uvs.append(uv)
        off += len(v)
    mesh_object("bonsai_wood", np.vstack(verts), np.vstack(faces), uv=np.vstack(uvs), mat=M["bark"])
    coll = new_collection("tuft", link=False)
    ob = mesh_object("tuft_shape", B["tuft_verts"] * MM, B["tuft_faces"], mat=M["needles"], coll=coll, smooth=True)
    T = B["tufts"]
    rot = []
    for x, y, z, axx, axy, axz, size, twist in T:
        q = Vector((axx, axy, axz)).to_track_quat("Z", "Y")
        q = q @ Euler((0, 0, float(twist))).to_quaternion()
        rot.append(tuple(q.to_euler()))
    points = np.array([tuple(gd.V(x, y, z)) for x, y, z in T[:, :3]])
    return instance_cloud("bonsai_needles", points, np.array(rot), T[:, 6], coll)


def instance_cloud(name, points, rot, scale, coll):
    """A vertex cloud whose points carry rotation and scale, instancing the collection's objects."""
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(points))
    me.vertices.foreach_set("co", np.asarray(points, np.float32).ravel())
    a = me.attributes.new("rot", "FLOAT_VECTOR", "POINT")
    a.data.foreach_set("vector", np.asarray(rot, np.float32).ravel())
    b = me.attributes.new("scl", "FLOAT", "POINT")
    b.data.foreach_set("value", np.asarray(scale, np.float32).ravel())
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ng = bpy.data.node_groups.new(name + "_nodes", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    N = Nodes(ng)
    gi, go = N.new("NodeGroupInput"), N.new("NodeGroupOutput")
    inst = N.new("GeometryNodeInstanceOnPoints")
    ci = N.new("GeometryNodeCollectionInfo", transform_space="ORIGINAL")
    ci.inputs["Collection"].default_value = coll
    ci.inputs["Separate Children"].default_value = True
    ci.inputs["Reset Children"].default_value = True
    inst.inputs["Pick Instance"].default_value = True
    rnd = N.new("FunctionNodeRandomValue", data_type="INT")
    N.sock(rnd.inputs, "Max", "INT").default_value = len(coll.objects) - 1
    N.link(N.sock(rnd.outputs, "Value", "INT"), inst.inputs["Instance Index"])
    r = N.new("GeometryNodeInputNamedAttribute", data_type="FLOAT_VECTOR")
    r.inputs["Name"].default_value = "rot"
    e2r = N.new("FunctionNodeEulerToRotation")
    N.link(N.sock(r.outputs, "Attribute", "VECTOR"), e2r.inputs["Euler"])
    N.link(e2r.outputs[0], inst.inputs["Rotation"])
    sc = N.new("GeometryNodeInputNamedAttribute", data_type="FLOAT")
    sc.inputs["Name"].default_value = "scl"
    N.link(N.sock(sc.outputs, "Attribute", "VALUE"), inst.inputs["Scale"])
    N.link(gi.outputs[0], inst.inputs["Points"])
    N.link(ci.outputs[0], inst.inputs["Instance"])
    N.link(inst.outputs[0], go.inputs[0])
    ob.modifiers.new("instances", "NODES").node_group = ng
    return ob


def moss_shoots(M, rng):
    """Moss, mm-scale (metres out). Living cushion moss near the water: short upright shoots with
    leaves spreading into a rosette at the top. Preserved sheet moss: flat feathery fronds."""
    colls = {}
    for name in ("living", "preserved"):
        coll = new_collection("moss_" + name, link=False)
        colls[name] = coll
        for k in range(5):
            verts, faces = [], []
            if name == "living":
                h = rng.uniform(0.8, 1.6)
                nleaf = int(rng.integers(14, 21))
                for j in range(nleaf):
                    t = (j + 1) / nleaf
                    c = np.array([0.0, 0.0, h * t])
                    yaw = j * 2.39996
                    L = 0.45 + 0.45 * t
                    lift = 1.2 - 0.9 * t                        # low leaves hug the stem, top ones spread
                    d = np.array([math.cos(yaw), math.sin(yaw), lift])
                    d /= np.linalg.norm(d)
                    side = np.array([-math.sin(yaw), math.cos(yaw), 0]) * 0.11
                    i = len(verts)
                    verts += [c - side, c + side, c + d * L + side * 0.4, c + d * L * 1.15, c + d * L - side * 0.4]
                    faces += [(i, i + 1, i + 2), (i, i + 2, i + 4), (i + 4, i + 2, i + 3)]
                i = len(verts)
                verts += [[-0.06, 0, 0], [0.06, 0, 0], [0.04, 0, h], [-0.04, 0, h]]
                faces += [(i, i + 1, i + 2), (i, i + 2, i + 3)]
            else:
                L = rng.uniform(1.5, 2.6)
                rise = rng.uniform(0.06, 0.22)
                for j in range(20):
                    t = j / 19
                    c = np.array([L * t, 0.0, 0.2 + rise * L * t * (1 - 0.5 * t)])
                    for sgn in (-1, 1):
                        d = np.array([0.4, sgn * 0.9, 0.12])
                        d /= np.linalg.norm(d)
                        leaf = 0.5 * (1 - 0.5 * t) + 0.18
                        i = len(verts)
                        verts += [c, c + np.array([0.12, 0, 0]), c + d * leaf]
                        faces.append((i, i + 1, i + 2))
                i = len(verts)
                verts += [[0, -0.05, 0.25], [0, 0.05, 0.25], [L, 0.03, 0.25 + rise * L * 0.5], [L, -0.03, 0.25 + rise * L * 0.5]]
                faces += [(i, i + 1, i + 2), (i, i + 2, i + 3)]
            ob = mesh_object(f"shoot_{name}_{k}", np.array(verts) * MM, faces, mat=M["moss_" + name], coll=coll, smooth=False)
    return colls["living"], colls["preserved"]


def moss_on(ob, coll, density_per_mm2, seed, clump_scale=55.0):
    """Instance moss over the ground mesh: denser and taller in the middle of cushions (a noise
    field), each shoot turned at random about the surface normal."""
    ng = bpy.data.node_groups.new(ob.name + "_moss", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    N = Nodes(ng)
    gi, go = N.new("NodeGroupInput"), N.new("NodeGroupOutput")
    pos = N.new("GeometryNodeInputPosition")
    clump = N.new("ShaderNodeTexNoise")
    N.set(clump, Scale=clump_scale, Detail=3.0, Roughness=0.55)
    N.link(pos.outputs[0], clump.inputs["Vector"])
    dens = N.new("ShaderNodeMapRange")
    N.set(dens, From_Min=0.3, From_Max=0.7, To_Min=0.25 * density_per_mm2 * 1e6, To_Max=1.5 * density_per_mm2 * 1e6)
    N.link(clump.outputs["Fac"], dens.inputs["Value"])
    dist = N.new("GeometryNodeDistributePointsOnFaces", distribute_method="RANDOM")
    dist.inputs["Seed"].default_value = seed
    N.link(dens.outputs["Result"], dist.inputs["Density"])
    N.link(gi.outputs[0], dist.inputs["Mesh"])
    ci = N.new("GeometryNodeCollectionInfo", transform_space="ORIGINAL")
    ci.inputs["Collection"].default_value = coll
    ci.inputs["Separate Children"].default_value = True
    ci.inputs["Reset Children"].default_value = True
    inst = N.new("GeometryNodeInstanceOnPoints")
    inst.inputs["Pick Instance"].default_value = True
    rnd = N.new("FunctionNodeRandomValue", data_type="INT")
    N.sock(rnd.inputs, "Max", "INT").default_value = len(coll.objects) - 1
    N.link(N.sock(rnd.outputs, "Value", "INT"), inst.inputs["Instance Index"])
    twist = N.new("FunctionNodeRandomValue", data_type="FLOAT_VECTOR")
    N.sock(twist.inputs, "Min", "VECTOR").default_value = (-0.3, -0.3, 0)
    N.sock(twist.inputs, "Max", "VECTOR").default_value = (0.3, 0.3, 6.283)
    e2r = N.new("FunctionNodeEulerToRotation")
    N.link(N.sock(twist.outputs, "Value", "VECTOR"), e2r.inputs["Euler"])
    rot = N.new("FunctionNodeRotateRotation", rotation_space="LOCAL")
    N.link(dist.outputs["Rotation"], rot.inputs["Rotation"])
    N.link(e2r.outputs[0], rot.inputs["Rotate By"])
    N.link(rot.outputs[0], inst.inputs["Rotation"])
    sc = N.new("FunctionNodeRandomValue", data_type="FLOAT")
    N.sock(sc.inputs, "Min", "VALUE").default_value = 0.7
    N.sock(sc.inputs, "Max", "VALUE").default_value = 1.2
    tall = N.new("ShaderNodeMapRange")
    N.set(tall, From_Min=0.3, From_Max=0.7, To_Min=0.65, To_Max=1.35)
    N.link(clump.outputs["Fac"], tall.inputs["Value"])
    N.link(N.math("MULTIPLY", N.sock(sc.outputs, "Value", "VALUE"), tall.outputs["Result"]), inst.inputs["Scale"])
    N.link(dist.outputs["Points"], inst.inputs["Points"])
    N.link(ci.outputs[0], inst.inputs["Instance"])
    join = N.new("GeometryNodeJoinGeometry")
    N.link(inst.outputs[0], join.inputs[0])
    N.link(gi.outputs[0], join.inputs[0])
    N.link(join.outputs[0], go.inputs[0])
    ob.modifiers.new("moss", "NODES").node_group = ng


def build_arm(gd: Garden, M):
    A, Hd = gd.s["arm"], gd.s["head"]
    bx, by = A["base"]
    q1, q2, z, q4 = A["pose"]
    a1 = q1 + math.pi
    a12 = a1 + q2
    hdg = a12 + q4
    ex, ey = A["elbow"]
    tx, ty = A["tool"][0], A["tool"][1]
    LH = A["link_height"]
    floor = -gd.bed
    col = prism("column", gd, bx, by, floor, LH, 44, 38, 48, M["walnut"])
    col.modifiers.clear()
    uv_cyl(col)
    prism("column_ring", gd, bx, by, LH - 5, LH - 1, 40, 40, 48, M["brass"])

    def slab(name, x0, y0, x1, y1, width, z0, thick, mat):
        r = width / 2
        ang = math.atan2(y1 - y0, x1 - x0)
        L = math.dist((x0, y0), (x1, y1))
        outline = [(x0 + r * math.cos(a), y0 + r * math.sin(a)) for a in np.linspace(ang + math.pi / 2, ang + 3 * math.pi / 2, 16)]
        outline += [(x1 + r * math.cos(a), y1 + r * math.sin(a)) for a in np.linspace(ang - math.pi / 2, ang + math.pi / 2, 16)]
        n = len(outline)
        verts = [gd.V(x, y, z0) for x, y in outline] + [gd.V(x, y, z0 + thick) for x, y in outline]
        faces = [(k, (k + 1) % n, n + (k + 1) % n, n + k) for k in range(n)]
        uv = [((math.cos(ang) * (x - x0) + math.sin(ang) * (y - y0)) / 280, (-math.sin(ang) * (x - x0) + math.cos(ang) * (y - y0)) / 280 + h * 0.2)
              for h in (0, 1) for x, y in outline]
        ob = mesh_object(name, verts, faces, uv=uv, mat=mat, smooth=False)
        bm = bmesh.new()
        bm.from_mesh(ob.data)
        bm.verts.ensure_lookup_table()
        bm.faces.new([bm.verts[k] for k in reversed(range(n))])
        bm.faces.new([bm.verts[n + k] for k in range(n)])
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(ob.data)
        bm.free()
        uvl = ob.data.uv_layers[0]
        for poly in ob.data.polygons:                               # caps: planar UVs along the link
            if len(poly.vertices) > 4:
                for li in poly.loop_indices:
                    co = ob.data.vertices[ob.data.loops[li].vertex_index].co
                    xx, yy = co.x / MM + gd.W / 2, co.y / MM + gd.D / 2
                    uvl.data[li].uv = ((math.cos(ang) * (xx - x0) + math.sin(ang) * (yy - y0)) / 280,
                                       (-math.sin(ang) * (xx - x0) + math.cos(ang) * (yy - y0)) / 280)
        bev = ob.modifiers.new("soft", "BEVEL")
        bev.width, bev.segments, bev.limit_method = 1.5 * MM, 3, "ANGLE"
        return ob
    slab("link1", bx, by, ex, ey, A["w1"], LH, 28, M["walnut"])
    prism("cap1", gd, bx, by, LH + 28, LH + 34, 20, 20, 32, M["brass"])
    slab("link2", ex, ey, tx, ty, A["w2"], LH + 34, 22, M["walnut"])
    prism("elbow", gd, ex, ey, LH, LH + 36, 24, 24, 32, M["brass"])
    prism("sleeve", gd, tx, ty, LH + 30, LH + 62, 15, 15, 32, M["walnut"])
    sole = 0.0                                                    # released and low: the skid rides the sand
    prism("rod", gd, tx, ty, sole + 21, LH + 60, 4.5, 4.5, 16, M["steel"])
    ux, uy = math.cos(hdg), math.sin(hdg)
    vx, vy = -uy, ux

    def block(name, cu, cv_, cz, su, sv, sz, mat):
        corners = [(cu + du * su / 2, cv_ + dv * sv / 2) for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        pts = [(tx + ux * a + vx * b, ty + uy * a + vy * b) for a, b in corners]
        verts = [gd.V(x, y, cz - sz / 2) for x, y in pts] + [gd.V(x, y, cz + sz / 2) for x, y in pts]
        faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        ob = mesh_object(name, verts, faces, mat=mat, smooth=False, uv=[(v[0] * 3, v[1] * 3) for v in verts])
        ob.modifiers.new("soft", "BEVEL").width = 0.6 * MM
        return ob
    half = (Hd["tines"] - 1) / 2 * Hd["pitch"] + Hd["tine_width"] / 2
    block("head", 0, 0, sole + 13, 10, 2 * half + 4, 16, M["walnut"])
    block("blade", -7, 0, sole + 5.5, 2.5, Hd["blade"], 11, M["steel"])
    block("skid", 17, 0, sole + 1, 22, 16, 2, M["brass"])
    for k in range(Hd["tines"]):
        off = (k - (Hd["tines"] - 1) / 2) * Hd["pitch"]
        x, y = tx + vx * off, ty + vy * off
        prism(f"tine_{k}", gd, x, y, -Hd["depth"] * 0.6, sole + 12, Hd["tine_width"] / 2, Hd["tine_width"] / 2, 12, M["steel"])


def uv_cyl(ob):
    me = ob.data
    uvl = me.uv_layers.new(name="UVMap")
    c = sum((v.co for v in me.vertices), Vector()) / len(me.vertices)
    for poly in me.polygons:
        for li in poly.loop_indices:
            co = me.vertices[me.loops[li].vertex_index].co
            a = math.atan2(co.y - c.y, co.x - c.x)
            uvl.data[li].uv = (co.z / 0.28, a * 0.041 / 0.28)       # grain along the column's length


# ------------------------------------------------------------------ light, camera, render
def setup_render(q, device, samples):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    cy = sc.cycles
    cy.samples = samples
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold = 0.015
    cy.use_denoising = True
    cy.denoiser = "OPENIMAGEDENOISE"
    cy.max_bounces, cy.diffuse_bounces, cy.glossy_bounces = 10, 4, 4
    cy.transmission_bounces, cy.transparent_max_bounces = 10, 12
    cy.caustics_reflective = cy.caustics_refractive = False
    cy.blur_glossy = 1.0
    sc.render.resolution_x, sc.render.resolution_y = q["res"]
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_depth = "8"
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast" if "AgX - Medium High Contrast" in [
        e.identifier for e in sc.view_settings.bl_rna.properties["look"].enum_items] else "None"
    if device == "GPU":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for kind in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
            try:
                prefs.compute_device_type = kind
                prefs.get_devices()
                if any(d.type == kind for d in prefs.devices):
                    break
            except TypeError:
                continue
        for d in prefs.devices:
            d.use = d.type != "CPU"
        cy.device = "GPU"
        print("GPU:", prefs.compute_device_type, [d.name for d in prefs.devices if d.use], flush=True)
    else:
        cy.device = "CPU"


def world_day():
    w = bpy.data.worlds.get("day") or bpy.data.worlds.new("day")
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    N = Nodes(nt)
    # a room by a window: the sky's gradient, but desaturated and warmed by the walls it bounces off,
    # so shadows fill neutral-warm as indoors instead of going blue as on a balcony
    sky = N.new("ShaderNodeTexSky", sky_type="MULTIPLE_SCATTERING")
    sky.sun_disc = False
    sky.sun_elevation, sky.sun_rotation = math.radians(32), math.radians(230)
    hs = N.new("ShaderNodeHueSaturation")
    N.set(hs, Saturation=0.3, Value=1.0)
    N.link(sky.outputs[0], hs.inputs["Color"])
    warm = N.mix(0.45, (0, 0, 0), (0.62, 0.56, 0.48))
    tint = N.new("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
    tint.inputs[0].default_value = 1.0
    N.link(hs.outputs["Color"], tint.inputs[6])
    N.link(warm, tint.inputs[7])
    room = N.new("ShaderNodeMix", data_type="RGBA", blend_type="ADD")
    room.inputs[0].default_value = 1.0
    N.link(tint.outputs[2], room.inputs[6])
    room.inputs[7].default_value = (0.06, 0.055, 0.048, 1)
    bg = N.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = 1.6
    N.link(room.outputs[2], bg.inputs["Color"])
    N.link(bg.outputs[0], N.new("ShaderNodeOutputWorld").inputs["Surface"])
    return w


def world_evening(gd: Garden):
    """The lux model's evening room light: an even glow from above (a lit ceiling), none from below,
    so a level surface gets ambient_lux and a wall half of it, as in karesansui.render.bake."""
    ev = gd.s["evening"]
    w = bpy.data.worlds.get("evening") or bpy.data.worlds.new("evening")
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    N = Nodes(nt)
    sep = N.new("ShaderNodeSeparateXYZ")
    N.link(N.new("ShaderNodeTexCoord").outputs["Generated"], sep.inputs[0])       # the ray direction
    bg = N.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (*ev["ambient_rgb"], 1)
    radiance = ev["ambient_lux"] * LUX_TO_W / math.pi                            # E = pi L on a level surface
    N.link(N.math("MULTIPLY", N.math("GREATER_THAN", sep.outputs["Z"], 0.0), radiance), bg.inputs["Strength"])
    N.link(bg.outputs[0], N.new("ShaderNodeOutputWorld").inputs["Surface"])
    return w


def glare():
    """Veiling glare of the lens round the lantern windows, which are ~11 stops brighter than the
    sand the evening is exposed for (compositor, scene-linear, before the view transform)."""
    ng = bpy.data.node_groups.get("glare")
    if ng is None:
        ng = bpy.data.node_groups.new("glare", "CompositorNodeTree")
        ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        gl = ng.nodes.new("CompositorNodeGlare")
        for k, v in (("Type", "Fog Glow"), ("Threshold", 0.25), ("Strength", 0.35), ("Size", 0.5)):
            gl.inputs[k].default_value = v
        ng.links.new(ng.nodes.new("CompositorNodeRLayers").outputs["Image"], gl.inputs["Image"])
        ng.links.new(gl.outputs["Image"], ng.nodes.new("NodeGroupOutput").inputs[0])
    return ng


LUX_TO_W = 1 / 25000.0            # scene irradiance (W/m2) per lux: sun strength 4 ~ 100 klx


def lighting(gd, mode, lights, sun):
    sc = bpy.context.scene
    vs = sc.view_settings
    if mode == "day":
        sc.world = world_day()
        sun.hide_render = False
        vs.exposure = 0.0
        vs.use_white_balance = False
        sc.render.use_compositing = False
    else:
        sc.world = world_evening(gd)
        sun.hide_render = True
        # sunlit sand (~100 klx) to lantern-lit sand (~75 lux) is ~10.4 stops: expose for the lanterns,
        # white-balanced to the room light as a camera would be, so the 2200 K lanterns stay amber
        vs.exposure = 10.2
        vs.use_white_balance, vs.white_balance_temperature = True, gd.s["evening"]["ambient_cct"]
        sc.compositing_node_group = glare()
        sc.render.use_compositing = True
    for paper, radiance in lights:
        # lit for the evening; by day the lanterns are off and the windows are plain paper
        paper.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = radiance if mode == "evening" else 0.0


def add_sun(gd):
    sd = bpy.data.lights.new("sun", "SUN")
    sd.energy = 4.0
    sd.angle = math.radians(1.2)
    sd.use_temperature, sd.temperature = True, 5400.0
    ob = bpy.data.objects.new("sun", sd)
    to_sun = Vector((-0.8, 0.4, 0.44)).normalized()                # from the back left, low: raking light
    ob.rotation_euler = (-to_sun).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.collection.objects.link(ob)
    fill = bpy.data.lights.new("window", "AREA")                      # soft bounce from the room, front right
    fill.shape, fill.size, fill.size_y = "RECTANGLE", 1.6, 1.0
    fill.energy = 10.0
    fill.use_temperature, fill.temperature = True, 6500.0
    fo = bpy.data.objects.new("window", fill)
    fo.location = gd.V(1250, -900, 900)
    fo.rotation_euler = (gd.V(350, 225, 0) - fo.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.collection.objects.link(fo)
    return ob, fo


VIEWS = {   # camera position, target (garden mm), lens (mm), f-stop, focus target
    "front": dict(pos=(350, -1000, 760), target=(350, 250, 75), lens=38, fstop=11, focus=(370, 225, 20)),
    "evening": dict(pos=(350, -1000, 760), target=(350, 250, 75), lens=38, fstop=5.6, focus=(370, 225, 20)),
    "stream": dict(pos=(350, 0, 250), target=(120, 255, 25), lens=50, fstop=8, focus=(150, 250, 20)),
    "sand": dict(pos=(318, 36, 58), target=(372, 250, 6), lens=55, fstop=18, focus=(360, 195, 3)),
    "arm": dict(pos=(860, -250, 430), target=(470, 250, 80), lens=50, fstop=8, focus=(420, 330, 30)),
    "tree": dict(pos=(370, 60, 300), target=(85, 385, 175), lens=42, fstop=8, focus=(80, 390, 150)),
}


def camera(gd, view):
    v = VIEWS[view]
    cd = bpy.data.cameras.get("cam") or bpy.data.cameras.new("cam")
    ob = bpy.data.objects.get("cam") or bpy.data.objects.new("cam", cd)
    if ob.name not in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.link(ob)
    cd.lens, cd.sensor_width = v["lens"], 36.0
    cd.clip_start, cd.clip_end = 0.01, 30.0
    ob.location = gd.V(*v["pos"])
    ob.rotation_euler = (gd.V(*v["target"]) - ob.location).to_track_quat("-Z", "Y").to_euler()
    cd.dof.use_dof = True
    cd.dof.aperture_fstop = v["fstop"]
    cd.dof.focus_distance = (gd.V(*v["focus"]) - ob.location).length
    bpy.context.scene.camera = ob


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="out/render_scene")
    ap.add_argument("--out", default="out/renders")
    ap.add_argument("--views", default="front")
    ap.add_argument("--quality", default="preview", choices=list(QUALITY))
    ap.add_argument("--device", default="CPU", choices=["CPU", "GPU"])
    ap.add_argument("--samples", type=int)
    ap.add_argument("--res")
    ap.add_argument("--blend", help="also save the .blend here")
    a = ap.parse_args(argv)
    q = dict(QUALITY[a.quality])
    if a.res:
        q["res"] = tuple(int(v) for v in a.res.split("x"))
    samples = a.samples or q["samples"]
    t0 = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.unit_settings.system, sc.unit_settings.scale_length = "METRIC", 1.0
    gd = Garden(Path(a.scene))
    rng = np.random.default_rng(5)
    M = make_materials(q)
    build_tray(gd, M)
    ground = build_ground(gd, M)
    build_sand(gd, M)
    build_kerb(gd, M)
    build_stones_and_rocks(gd, M)
    build_cascades(gd, M)
    lights = []
    build_lanterns(gd, M, lights)
    build_bridge(gd, M)
    build_tree(gd, M)
    build_arm(gd, M)
    living, preserved = moss_shoots(M, rng)
    moss_on(ground["moss_living"], living, q["moss"], 1)
    moss_on(ground["moss_preserved"], preserved, q["moss"] * 0.85, 2)
    sun, fill = add_sun(gd)
    setup_render(q, a.device, samples)
    print(f"scene built in {time.time() - t0:.1f} s", flush=True)
    if a.blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(Path(a.blend).resolve()))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for view in a.views.split(","):
        mode = "evening" if view == "evening" else "day"
        lighting(gd, mode, lights, sun)
        fill.hide_render = mode != "day"
        camera(gd, view)
        sc.render.filepath = str((out / f"{view}.png").resolve())
        t1 = time.time()
        bpy.ops.render.render(write_still=True)
        print(f"{view}: {time.time() - t1:.0f} s -> {sc.render.filepath}", flush=True)


if __name__ == "__main__":
    # as a module (python render/garden_cycles.py ...) or inside Blender (blender -b -P ... -- ...)
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])
