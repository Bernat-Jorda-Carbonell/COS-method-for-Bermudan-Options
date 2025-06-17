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
