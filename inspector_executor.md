---
title: Inspector & Executor
layout: home
nav_order: 1
---

# Inspector

The inspector assigns sparse formats to matrix tiles and reorders data. We currently have support for dense, 2:4, and all-zero tiles. The tiles are then arranged into a tile-level Compressed Sparse Row (CSR) format.

<img src="media/inspector.svg"  alt="Inspector">

# Executor

A grid is launched with one Triton program per tile of output matrix D. Each program iterates over a row of A and column of B according to the schedule and accumulates the result for its tile.

<img src="media/executor.svg"  alt="Executor">