# AMS-SADE
Official MATLAB Implementation of AMS-SADE (Adaptive Multi-Swarm Surrogate-Assisted Differential Evolution)

## Key Innovations
Fast Local RBF Agents: Abandon global modeling to completely eliminate the $O(N^3)$ overhead induced by high-dimensional phantom effects.
Heterogeneous Multi-Swarm Co-evolution: Mitigate spatial divergence by coordinating sparse exploration, adaptive manifold contraction, and agent gradient guidance.
Dynamic Resource Allocation (DRA): Actively regulate the dynamic adjustment of computational resources via real-time evaluation feedback under strict 1-FE constraints.
## Usage
Simply import AMS_SADE.m into PlatEMO or your custom optimization framework as an independent algorithm module.
