# Robotic living zen garden

A desk-scale Japanese garden (moss, a pumped stream, a bonsai, stone lanterns and a raked
bed of fine white sand) with a robot arm built into it that erases and rakes the sand.

Status: designed and checked in silico. Nothing physical has been built yet.

- [`zen-garden/PLAN.md`](zen-garden/PLAN.md): the build plan. It covers the decisions and the evidence for them, phases with exit criteria, a rough bill of materials, risks, and what the model can't tell you.
- [`zen-garden/`](zen-garden/): the in-silico twin (Python). It holds the layout, rake patterns, a collision- and reach-checked planner for the arm, joint-space G-code for FluidNC with a strict checker, a sand simulation, lantern-lit renders and water-loop sizing. See its README for how to run it.
