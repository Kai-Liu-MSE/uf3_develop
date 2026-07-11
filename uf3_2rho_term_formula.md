# Density-Modulated Two-Body Term Used in UF3

This note records the density-modulated two-body term, denoted here as `2rho`,
used in the current Co UF3 models.

## Total Energy

The current model has the form

```text
E = E1 + E2 + E3 + E2rho
```

where `E2rho` is the added density-modulated two-body contribution.

## Local Density

For each atom `i`, the local scalar density is defined as

```text
rho_i = sum_j r_ij^(-1)
```

with the sum over neighboring Co atoms satisfying

```text
0.1 A < r_ij < 7.0 A
```

Equivalently, for the current single-element Co model,

```text
rho_i = sum_{j in Co, 0.1 < r_ij < 7.0} 1 / r_ij
```

The density is normalized by the corresponding bulk reference density:

```text
x_i = rho_i / rho_bulk
```

For the current `c7 n18` model,

```text
rho_bulk = 27.514329063150058
```

This value is estimated from an ideal hcp Co bulk environment using the same
density definition.

## Two Density Channels

The model uses two density-modulation channels:

```text
alpha = 0.5
alpha = 1.5
```

For each channel, the modulation function is the conservative form

```text
g_alpha(x) = x^alpha - 1 - alpha * (x - 1)
```

Thus the two channels are

```text
g_0.5(x) = x^0.5 - 1 - 0.5 * (x - 1)
```

and

```text
g_1.5(x) = x^1.5 - 1 - 1.5 * (x - 1)
```

This conservative form satisfies

```text
g_alpha(1) = 0
g_alpha'(1) = 0
```

Therefore, in a bulk-like environment where `rho_i = rho_bulk`, the `2rho`
term has no zeroth-order or first-order contribution with respect to the
normalized density deviation. It mainly responds to environments whose local
density differs from bulk, such as surfaces and defects.

## Density-Modulated Two-Body Energy

For each channel, the density-modulated two-body potential is represented by a
B-spline expansion:

```text
V_alpha(r) = sum_m c_{alpha,m} B_m(r)
```

The full `2rho` energy is then

```text
E2rho = sum_alpha sum_i sum_j g_alpha(x_i) * V_alpha(r_ij)
```

or, expanded in the spline basis,

```text
E2rho = sum_alpha sum_m c_{alpha,m}
        sum_i sum_j g_alpha(rho_i / rho_bulk) * B_m(r_ij)
```

where the pair-distance sum uses the same Co-Co pair cutoff as the ordinary
two-body term in the current model.

## Parameter Count

For the current `c7 n18` model, each density channel has 21 spline
coefficients:

```text
alpha = 0.5: 21 parameters
alpha = 1.5: 21 parameters
```

Thus the `2rho` term adds

```text
2 * 21 = 42 parameters
```

The full `c7 n18` model parameter count is

```text
1body Co       : 1
2body Co-Co    : 21
3body Co-Co-Co : 539
2rho           : 42
total          : 603
```

Without the `2rho` term, the corresponding model would have

```text
1 + 21 + 539 = 561 parameters
```

so `2rho` increases the parameter count by about

```text
42 / 561 = 7.5%
```

## Interpretation

The ordinary two-body term depends only on pair distance:

```text
E2 ~ V(r_ij)
```

The `2rho` term depends on both pair distance and the local density around the
central atom:

```text
E2rho ~ g_alpha(rho_i / rho_bulk) * V_alpha(r_ij)
```

Thus the same Co-Co distance can receive different corrections in bulk-like,
surface-like, and low-coordination environments.
