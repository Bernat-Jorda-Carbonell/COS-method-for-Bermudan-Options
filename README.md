
# COS Method for Bermudan Option Pricing

A fast, accurate implementation of the Fourier–cosine (COS) expansion method for
pricing Bermudan (and American) options under both the Black–Scholes and
jump-diffusion (CGMY) models.

## Overview

This project implements the COS method of Fang & Oosterlee (2009) to achieve
exponential convergence in option pricing. By combining an FFT-based backward
induction with Richardson extrapolation, we deliver sub-millisecond runtimes
with 9-digit accuracy on standard hardware.

- **Models supported**  
  - Black–Scholes (log-normal diffusion)  
  - CGMY exponential-Lévy (infinite-activity jumps)

- **Option types**  
  - Bermudan puts (extendable to calls)  
  - American-limit via extrapolation

- **Key strengths**  
  - Exponential convergence in series term count `N`  
  - O(M N log N) complexity (M = # of exercise dates)  
  - FFT core highly parallelizable  
  - 4-point Richardson extrapolation for American limit

## Features

- **Modular Python 3.10+ code**  
  - Characteristic-function routines  
  - COS coefficient computation  
  - FFT-accelerated backward induction  
  - Convergence analysis scripts

- **Benchmarks** reproduce  
  - Fang & Oosterlee’s convergence tables  
  - CONV-method comparisons

- **Notebooks** for  
  - Runtime vs. `N` and `M`  
  - Error‐decay plots  
  - Jump-diffusion parameter effects

- **Documentation**  
  - LaTeX math derivations  
  - Usage examples and CLI guide

## Installation

```bash
git clone \
  https://github.com/yourusername/cos-bermudan-options.git
cd cos-bermudan-options

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````

## Usage

1. **Price a Bermudan put under Black–Scholes**

   ```bash
   python price_cos.py \
     --model bs \
     --K 100 --S0 100 \
     --r 0.05 --sigma 0.2 \
     --T 1.0 --M 10 --N 256
   ```

2. **Switch to CGMY model**

   ```bash
   python price_cos.py \
     --model cgmy \
     --C 1.0 --G 5.0 \
     --M 0.5 --Y 0.7 \
     --T 1.0 --K 100 \
     --S0 100 --r 0.05 \
     --M 10 --N 512
   ```

3. **Run benchmarks**

   ```bash
   jupyter notebook \
     benchmarks/convergence.ipynb
   ```

## Performance

|   N |  M | Runtime (ms) | Max Error |
| --: | -: | -----------: | --------: |
| 128 | 10 |         12.3 |     <1e-6 |
| 256 | 20 |         18.7 |     <1e-8 |
| 512 | 50 |         45.2 |     <1e-9 |

## References

1. Fang, F., & Oosterlee, C. W. (2009). *A novel pricing method for European
   options based on Fourier–cosine series expansions.*
2. Ruijter, M. J., & Oosterlee, C. W. (2012). *Pricing early-exercise and discrete
   barrier options by Fourier-cosine series expansions.*

---

> *Developed for the Computational Finance course at Mastermath (netherlands)
> (June 2025).*

---
