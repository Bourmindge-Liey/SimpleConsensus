# Simple Consensus 

A wrapped multi-agent systems.

## Problem Definition

Each individual has the simpliest internal dynamics 
$$\dot{x}_i = u_i$$

The network is defined by vertices and edges with adjacency matrix.
$$\mathcal{G} = G(\mathcal{V, E, A})$$

Definition of neighbour:
$$\mathcal{N}_i = \{v_j \in \mathcal{V} : (v_i, v_j) \in \mathcal{E} \}$$
meaning $a_{ij} : i \rarr j$.


## Protocol

1. Basic Protocol

    For switching topology and no communication delay

$$ u_i = \sum^N_{j=1} a_{ij} (x_j - x_i) $$

$$ u = \begin{bmatrix} 
        a_{11} - \sum^{N}_{j=1}a_{1j}  &a_{12} &... &a_{1N} \\
        a_{21}   &a_{22}- \sum^{N}_{j=1}a_{2j} &... &a_{2N} \\
        ... &... &... &...\\
        a_{N1} &a_{N2} &... &a_{NN} - \sum^{N}_{j=1}a_{Nj}\\ 
    \end{bmatrix} x$$

$$ u = (A - \text{diag}(A \cdot \textbf{1})) x $$

## System
$$\dot{x} = -Lx $$
where  
$$ L = \text{diag}(A \cdot \textbf{1}) - A $$

## Disagreement

$$\Phi_G(x) = \frac{1}{2}x^T L x $$

## Consensus 

1. Average Consensus
$$ \alpha = \text{Avg}(x)$$

2. Max/Min Consensus 
$$ \alpha = \max | \min (x)$$
