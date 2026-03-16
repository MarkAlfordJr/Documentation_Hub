# Prototype Script Process

### Process
1. Import libraries
2. Problem Defining/Setup
3. MLP Building
4. PINN Model
5. Autodiff: derivatives needed for PED residual
6. Exact Solution (for evalution only)
7. Sampling Points
8. Loss Function
9. Optimzier
10. Training Loop
11. Evalution on a grid
12. Plotting training Loss

## Import Libraries
these are the libraries to import

```python
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import autograd
import optax
```





