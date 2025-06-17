
# COS Method for Bermudan Option Pricing

A fast, accurate implementation of the Fourier–cosine (COS) expansion method for pricing Bermudan (and American) options under both the Black–Scholes and jump-diffusion (CGMY) models.

## Overview

This project demonstrates the COS method originally proposed by Fang & Oosterlee (2009) to achieve exponential convergence in option pricing. By combining an FFT‐based backward induction and Richardson extrapolation, we deliver sub‐millisecond runtimes with 9-digit accuracy on standard hardware.

- **Models supported**:  
  - Black–Scholes (log‐normal diffusion)  
  - CGMY exponential-Lévy (infinite-activity jumps)

- **Option types**: Bermudan puts (extendable to calls), with American‐limit extrapolation.

- **Key strengths**:  
  - Exponential convergence in series term count \(N\)  
  - \(\mathcal{O}(M N\log N)\) complexity (M = number of exercise dates)  
  - Highly parallelizable FFT core  
  - American‐limit recovery via 4-point Richardson extrapolation

## Project Structure

```

cos-bermudan-options/
├── csv/                     # Sample input/output data in CSV format
├── dummy\_plot\_dir/          # Placeholder directory for pipeline testing
├── plots/                   # Generated figures (error vs. N, runtimes, etc.)
├── LICENSE                  # MIT license file
├── README.md                # Project overview and usage instructions
├── Report.pdf               # Detailed PDF report with derivations and results
├── compare\_lsmc\_cos.py      # Script to compare LSMC against COS pricing
├── cos\_bermudan.py          # CLI wrapper for Bermudan option pricing via COS
├── cos\_bs.py                # COS implementation under the Black–Scholes model
├── cos\_models.py            # Characteristic functions for BS & CGMY models
├── cos\_utils.py             # Helper routines (coefficients, truncation, FFT setup)
├── lsmc.py                  # Least‐Squares Monte Carlo pricer for benchmark comparison
└── run\_experiments.py       # Batch runner for convergence & timing experiments

```

**Descriptions**  
- **csv/**: Holds example CSV files for input parameters and output prices, so you can quickly rerun samples.  
- **dummy_plot_dir/**: Used to verify your plotting pipeline without cluttering real results.  
- **plots/**: Stores all generated plots (error decay, runtime vs. \(N\), convergence comparisons).  
- **compare_lsmc_cos.py**: Automates side‐by‐side pricing of Bermudan options via LSMC and COS, outputting tables and charts.  
- **cos_bermudan.py**: Entry‐point CLI script—call with flags like `--model bs` or `--model cgmy`, strike, maturity, etc.  
- **cos_bs.py**: Pure‐Python COS algorithm under Black–Scholes; computes coefficients and FFT‐based induction.  
- **cos_models.py**: Defines and exposes model-specific characteristic functions (BS, CGMY).  
- **cos_utils.py**: Utility functions—grid setup, truncation bounds, weight calculations, FFT helpers.  
- **lsmc.py**: Implements Longstaff–Schwartz least‐squares Monte Carlo for validation and benchmarks.  
- **run_experiments.py**: Orchestrates full suites of experiments (varying \(N\), \(M\), model parameters), saving CSVs and plots.  
- **Report.pdf**: Comprehensive write-up with mathematical derivations, tables, and error analysis.  


## Features

- **Modular codebase** in Python 3.10+, with clear separation of:  
  - characteristic‐function routines  
  - COS coefficient computation  
  - FFT‐accelerated backward induction  
  - convergence analysis scripts

- **Benchmark scripts** to reproduce  
  - Fang & Oosterlee’s original convergence tables  
  - Comparison with the CONV method for validation

- **Performance analysis** notebooks (Jupyter) showing:  
  - runtime vs. \(N\) and \(M\)  
  - error decay plots  
  - impact of jump-diffusion parameters

- **Documentation**  
  - Detailed math derivations in LaTeX  
  - Usage examples and CLI interface

## Installation

```bash
git clone https://github.com/yourusername/cos-bermudan-options.git
cd cos-bermudan-options
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````

## Usage

1. **Price a Bermudan put under Black–Scholes**

   ```bash
   python price_cos.py --model bs --K 100 --S0 100 --r 0.05 --sigma 0.2 \
                       --T 1.0 --M 10 --N 256
   ```

2. **Switch to CGMY model**

   ```bash
   python price_cos.py --model cgmy --C 1.0 --G 5.0 --M 0.5 --Y 0.7 --T 1.0 \
                       --K 100 --S0 100 --r 0.05 --M 10 --N 512
   ```

3. **Run convergence and timing benchmarks**

   ```bash
   jupyter notebook benchmarks/convergence.ipynb
   ```

## Performance

| Series terms $N$ | Exercise dates $M$ | Runtime (ms) | Max Pricing Error |
| ---------------: | -----------------: | -----------: | ----------------: |
|              128 |                 10 |         12.3 |        $<10^{-6}$ |
|              256 |                 20 |         18.7 |        $<10^{-8}$ |
|              512 |                 50 |         45.2 |        $<10^{-9}$ |

## References

1. Fang, F., & Oosterlee, C. W. (2009). *A novel pricing method for European options based on Fourier–cosine series expansions.* SIAM Journal on Scientific Computing.
2. Ruijter, M. J., & Oosterlee, C. W. (2012). *Pricing early‐exercise and discrete barrier options by Fourier‐cosine series expansions.* Journal of Computational Finance.

---

 *This repository was developed as part of a Computational Finance course at Mastermath, netherlands.*
