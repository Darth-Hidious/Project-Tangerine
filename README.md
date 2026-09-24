# Tangerine

A robotic living zen garden.

A desk-scale Japanese garden (moss, a pumped stream, a bonsai, stone lanterns and a raked
bed of fine white sand) with a robot arm built into it that erases and rakes the sand.

Status: designed and checked in silico. Nothing physical has been built yet.

- [`zen-garden/PLAN.md`](zen-garden/PLAN.md): the build plan. It covers the decisions and the evidence for them, phases with exit criteria, risks, what's ready for CAD, and what the model can't tell you.
- [`zen-garden/BOM.md`](zen-garden/BOM.md): the bill of materials, line by line, with the quantities computed from the model.
- [`zen-garden/`](zen-garden/): the in-silico twin (Python). It holds the layout, rake patterns, a collision- and reach-checked planner for the arm, joint-space G-code for FluidNC with a strict checker, a sand simulation, lantern-lit renders and water-loop sizing. It also has a 3D viewer and photoreal Blender Cycles renders of the same model (locally or on a Colab GPU). See its README for how to run it.

![Photoreal render of the garden, built from the model](zen-garden/docs/renders/front.jpg)

## Website

`vercel.json` turns the repository into a static site: the 3D viewer is the home page, and the renders are under `/renders/`. To publish it, import the repository in Vercel. The build needs no settings, and Vercel redeploys on every push to its production branch.
