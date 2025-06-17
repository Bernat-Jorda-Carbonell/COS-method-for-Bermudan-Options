# experiments.py

import os
import argparse
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma

from cos_bs import cumulant_range
from cos_bermudan import price_bermudan_cos, price_barrier_cos
from cos_models import bs_charfn, cgmy_charfn, nig_charfn
from lsmc import price_american_lsmc_bs, price_american_lsmc_cgmy

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run parameter sweep for COS Bermudan pricing (BS or CGMY)."
    )
    parser.add_argument(
        "--model", type=str, default="BS", choices=["BS", "CGMY", "NIG"],
        help="Underlying model: 'BS', 'CGMY', or 'NIG'."
    )
    parser.add_argument(
        "--S0", type=float, default=100.0, help="Initial spot (default 100)."
    )
    parser.add_argument(
        "--K", type=float, default=100.0, help="Strike (default 100)."
    )
    parser.add_argument(
        "--r", type=float, default=0.05, help="Risk‐free rate (default 0.05)."
    )
    parser.add_argument(
        "--sigma", type=float, default=0.2,
        help="Volatility (BS only, default 0.2)."
    )
    parser.add_argument(
        "--C", type=float, default=1.0,
        help="CGMY C parameter (only if --model CGMY)."
    )
    parser.add_argument(
        "--G", type=float, default=5.0,
        help="CGMY G parameter (only if --model CGMY)."
    )
    parser.add_argument(
        "--Mjump", type=float, default=5.0,
        help="CGMY M parameter (only if --model CGMY)."
    )
    parser.add_argument(
        "--Y", type=float, default=1.5,
        help="CGMY Y parameter (only if --model CGMY)."
    )
    parser.add_argument(
        "--T", type=float, default=1.0,
        help="Maturity T (default 1.0)."
    )
    parser.add_argument(
        "--M", type=int, default=10,
        help="Number of exercise dates M (default 10)."
    )
    parser.add_argument(
        "--L", type=float, default=8.0,
        help="Width L for truncation [a,b] (default 8)."
    )
    parser.add_argument(
        "--N_list", type=str, default="32,64,128,256,512",
        help="Comma-separated list of N values (default: '32,64,128,256,512')."
    )
    parser.add_argument(
        "--option_type", type=str, choices=["put", "call"], default="put",
        help="Option type: 'put' or 'call'."
    )
    parser.add_argument(
        "--output_csv", type=str, default="results.csv",
        help="Output CSV filename."
    )
    parser.add_argument(
        "--do_richardson", action="store_true",
        help="Perform 4-point Richardson extrapolation to estimate an American put."
    )
    parser.add_argument(
        "--d_max", type=int, default=3,
        help="Max Richardson level d (so smallest N=2^d)."
    )
    parser.add_argument(
        "--plots_dir", type=str, default="plots",
        help="Directory for saving plots."
    )
    parser.add_argument(
    "--do_barrier", action="store_true",
    help="Run discretely monitored barrier option experiments."
    )
    parser.add_argument(
        "--barrier_type", type=str, choices=["up-and-out", "down-and-out"], default="up-and-out",
        help="Type of barrier: 'up-and-out' or 'down-and-out'."
    )
    parser.add_argument(
        "--barrier_level", type=float, default=120.0,
        help="Barrier level H (absolute, not log)."
    )
    parser.add_argument(
        "--M_barrier", type=int, default=12,
        help="Number of monitoring dates for barrier option (e.g. 12 for monthly)."
    )
    parser.add_argument(
        "--do_lsmc", action="store_true",
        help="Also run a Longstaff–Schwartz American‐put (BS) benchmark."
    )
    parser.add_argument(
        "--M_sims_lsmc", type=int, default=200_000,
        help="Number of Monte Carlo paths for LSMC (default: 200000)."
    )
    parser.add_argument(
        "--basis_deg", type=int, default=3,
        help="Polynomial basis degree for LSMC regression (default: 3)."
    )
    parser.add_argument(
        "--alpha", type=float, default=15.0,
        help="NIG alpha parameter (only if --model NIG)."
    )
    parser.add_argument(
        "--beta", type=float, default=-5.0,
        help="NIG beta parameter (only if --model NIG)."
    )
    parser.add_argument(
        "--delta", type=float, default=0.5,
        help="NIG delta parameter (only if --model NIG)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Build the CF function depending on the chosen model
    if args.model == "BS":
        # For BS, our CF needs (u, dt, r, sigma). We wrap bs_charfn in a lambda that also closes over sigma.
        charfn = lambda u, dt, r: bs_charfn(u, dt, r, args.sigma)
    elif args.model == "CGMY":
        charfn = lambda u, dt, r: cgmy_charfn(u, dt, r, args.C, args.G, args.Mjump, args.Y)
    else:  # args.model == "NIG"
        charfn = lambda u, dt, r: nig_charfn(u, dt, r, args.alpha, args.beta, args.delta)

    # 2) Compute the truncation interval [a, b] once (model‐specific)
    x0 = np.log(args.S0 / args.K)

    if args.model == "BS":
        a, b = cumulant_range(args.r, args.sigma, args.S0, args.K, args.T, L=args.L)

    elif args.model == "CGMY":
        C, G, Mjump, Y = args.C, args.G, args.Mjump, args.Y
        c2 = args.T * C * gamma(2 - Y) * ( Mjump**(Y - 2) + G**(Y - 2) )
        c4 = args.T * C * gamma(4 - Y) * ( Mjump**(Y - 4) + G**(Y - 4) )
        half_width = args.L * np.sqrt(c2 + np.sqrt(c4))
        a = x0 - half_width
        b = x0 + half_width

    else:  # args.model == "NIG"
        alpha, beta, delta = args.alpha, args.beta, args.delta
        # c1 (mean shift)
        c1 = args.T * delta * ( np.sqrt(alpha**2 - beta**2) 
                                 - np.sqrt(alpha**2 - (beta + 1.0)**2) )
        # c2 (variance)
        c2 = args.T * delta * alpha**2 / ((alpha**2 - beta**2)**1.5)
        # c4 (fourth cumulant)
        c4 = 3.0 * args.T * delta * alpha**4 / ((alpha**2 - beta**2)**3.5)
        half_width = args.L * np.sqrt(c2 + np.sqrt(c4))
        a = (c1 + x0) - half_width
        b = (c1 + x0) + half_width

    # 3) Create plots directory if needed
    os.makedirs(args.plots_dir, exist_ok=True)

    # 4.1) If Barrier  is requested, perform it and exit
    if args.do_barrier:
        # Create a small loop over two monitoring frequencies: monthly (M_barrier=12) and daily (M_barrier=252)
        # and over a list of N values (e.g. N = [ 2**10, 2**11, 2**12, 2**13, 2**14 ] ).

        Ns = [2**10, 2**11, 2**12, 2**13, 2**14]
        results_barrier = []

        for Mbar in [12, 252]:
            for N in Ns:
                t0 = time.time()
                price_b = price_barrier_cos(
                    S0=args.S0,
                    K=args.K,
                    r=args.r,
                    T=args.T,
                    M=Mbar,                # monitoring frequency
                    N=N,
                    a=a, b=b,
                    charfn=charfn,
                    option_type=args.option_type,
                    barrier_type=args.barrier_type,
                    barrier_level=args.barrier_level
                )
                t1 = time.time()
                runtime = (t1 - t0) * 1000.0  # ms

                # If you have a reference barrier price V_ref (from a very large N or a separate method), compute error
                # For now, we can just record price and runtime.
                results_barrier.append({
                    "model": args.model,
                    "barrier_type": args.barrier_type,
                    "monitoring_freq": Mbar,
                    "N": N,
                    "price": price_b,
                    "runtime_ms": runtime
                })

                print(f"[Barrier {args.barrier_type}, M={Mbar}] N={N}, price={price_b:.6f}, time={runtime:.1f} ms")

        df_bar = pd.DataFrame(results_barrier)
        df_bar.to_csv(args.output_csv, index=False)
        print(f"Results saved to {args.output_csv}")
        
        

        # Plot price vs N for each monitoring frequency on two subplots:
        for Mbar in [12, 252]:
            df_sub = df_bar[df_bar["monitoring_freq"] == Mbar]
            plt.figure(figsize=(6,4))
            plt.plot(df_sub["N"], df_sub["price"], marker="o", linestyle="-")
            plt.xscale("log", base=2)
            plt.xlabel("COS terms (N)")
            plt.ylabel("Barrier Price")
            plt.title(f"Barrier Price (type={args.barrier_type}, M={Mbar}) vs N")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.savefig(os.path.join(args.plots_dir, f"barrier_price_M{Mbar}.png"), bbox_inches="tight")
            plt.close()

            plt.figure(figsize=(6,4))
            plt.plot(df_sub["N"], df_sub["runtime_ms"], marker="o", linestyle="-", color="C1")
            plt.xscale("log", base=2)
            plt.yscale("log")
            plt.xlabel("COS terms (N)")
            plt.ylabel("Runtime (ms)")
            plt.title(f"Barrier Runtime (type={args.barrier_type}, M={Mbar}) vs N")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.savefig(os.path.join(args.plots_dir, f"barrier_runtime_M{Mbar}.png"), bbox_inches="tight")
            plt.close()

        return   # skip the normal Bermudan or Richardson branches

    # 4.2) If Richardson extrapolation is requested, perform it and exit
    if args.do_richardson:
        records = []
        for d in range(args.d_max + 1):
            N0 = 2 ** d
            N1 = 2 ** (d + 1)
            N2 = 2 ** (d + 2)
            N3 = 2 ** (d + 3)

            # Price at N0
            t0 = time.time()
            v0, _ = price_bermudan_cos(
                S0=args.S0,
                K=args.K,
                r=args.r,
                T=args.T,
                M=args.M,
                N=N0,
                a=a,
                b=b,
                charfn=charfn,
                option_type=args.option_type
            )
            t1 = time.time()
            runtime0 = (t1 - t0) * 1000.0  # ms

            # Price at N1
            t0 = time.time()
            v1, _ = price_bermudan_cos(
                S0=args.S0,
                K=args.K,
                r=args.r,
                T=args.T,
                M=args.M,
                N=N1,
                a=a,
                b=b,
                charfn=charfn,
                option_type=args.option_type
            )
            t1 = time.time()
            runtime1 = (t1 - t0) * 1000.0

            # Price at N2
            t0 = time.time()
            v2, _ = price_bermudan_cos(
                S0=args.S0,
                K=args.K,
                r=args.r,
                T=args.T,
                M=args.M,
                N=N2,
                a=a,
                b=b,
                charfn=charfn,
                option_type=args.option_type
            )
            t1 = time.time()
            runtime2 = (t1 - t0) * 1000.0

            # Price at N3
            t0 = time.time()
            v3, _ = price_bermudan_cos(
                S0=args.S0,
                K=args.K,
                r=args.r,
                T=args.T,
                M=args.M,
                N=N3,
                a=a,
                b=b,
                charfn=charfn,
                option_type=args.option_type
            )
            t1 = time.time()
            runtime3 = (t1 - t0) * 1000.0

            # 4-point Richardson extrapolation:
            v_AM = (64.0 * v3 - 56.0 * v2 + 14.0 * v1 - v0) / 21.0

            records.append({
                "d": d,
                "N0": N0, "v0": v0, "time0_ms": runtime0,
                "N1": N1, "v1": v1, "time1_ms": runtime1,
                "N2": N2, "v2": v2, "time2_ms": runtime2,
                "N3": N3, "v3": v3, "time3_ms": runtime3,
                "v_AM": v_AM
            })

            print(
                f"[{args.model}] d={d}: N3={N3}, v_AM={v_AM:.6f}, "
                f"runtimes=({runtime0:.1f}, {runtime1:.1f}, {runtime2:.1f}, {runtime3:.1f}) ms"
            )

        df_R = pd.DataFrame(records)
        df_R.to_csv(args.output_csv, index=False)
        print(f"Results saved to {args.output_csv}")

        # Plot: extrapolated American price vs. d
        plt.figure(figsize=(6, 4))
        plt.plot(df_R["d"], df_R["v_AM"], marker="o", linestyle="-")
        plt.xlabel("Richardson level d")
        plt.ylabel("Extrapolated American Price v_AM")
        plt.title(f"American Put (extrapolated) vs d (model={args.model})")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.savefig(
            os.path.join(args.plots_dir, "richardson_vAM_vs_d.png"),
            bbox_inches="tight"
        )
        plt.close()

        # Plot: runtime (for v3) vs. N3
        plt.figure(figsize=(6, 4))
        plt.plot(df_R["N3"], df_R["time3_ms"], marker="o", linestyle="-", color="C1")
        plt.xscale("log", base=2)
        plt.yscale("log")
        plt.xlabel("COS terms (N3 = 2^{d+3})")
        plt.ylabel("Runtime (ms) for v(2^{d+3})")
        plt.title(f"Runtime vs N (Richardson) (model={args.model})")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.savefig(
            os.path.join(args.plots_dir, "richardson_runtime_vs_N3.png"),
            bbox_inches="tight"
        )
        plt.close()

        return  # Skip the normal N-list loop
    


   # 4.3) If Longstaff–Schwartz is requested, run it and exit
    if args.do_lsmc:

        # Unpack params
        S0, K, r, sigma, T = args.S0, args.K, args.r, args.sigma, args.T
        M_steps = args.M
        deg      = args.basis_deg

        # 1) Pick a high-precision COS reference once
        N_ref = 4096
        V_COS, _ = price_bermudan_cos(
            S0=S0, K=K, r=r, T=T, M=M_steps,
            N=N_ref, a=a, b=b, charfn=charfn,
            option_type=args.option_type
        )
        print(f"[COS-REF] N={N_ref}, price={V_COS:.6f}")

        # 2) sweep over a list of M_sims values
        M_sims_list = [500, 1000, 2000, 5000, 10000]
        records = []

        for M_sims in M_sims_list:
            t0 = time.time()
            if args.model == "BS":
                price_lsmc = price_american_lsmc_bs(
                    S0=S0, K=K, r=r, sigma=sigma,
                    T=T, M_steps=M_steps,
                    M_sims=M_sims,
                    basis_degree=deg,
                    seed=12345
                )
            else:  # CGMY or NIG
                price_lsmc = price_american_lsmc_cgmy(
                    S0=S0, K=K, r=r,
                    C=args.C, G=args.G,
                    M_jump=args.Mjump, Y=args.Y,
                    T=T, M_steps=M_steps,
                    M_sims=M_sims,
                    basis_degree=deg,
                    seed=12345
                )
            t1 = time.time()
            runtime_lsmc = (t1 - t0) * 1000.0
            abs_err = abs(price_lsmc - V_COS)

            print(
                f"[LSMC-{args.model}] M_sims={M_sims}, deg={deg}, "
                f"price={price_lsmc:.6f}, error={abs_err:.6f}, "
                f"time={runtime_lsmc:.1f} ms"
            )

            records.append({
                "M_sims": M_sims,
                "basis_deg": deg,
                "lsmc_price": price_lsmc,
                "abs_error": abs_err,
                "runtime_ms": runtime_lsmc
            })

        df_all = pd.DataFrame(records)
        df_all.to_csv(os.path.join("csv", "all_lsmc.csv"), index=False)
        print("Wrote csv/all_lsmc.csv")
        return
    

    # 5) Otherwise, run the normal sweep over N_list
    N_values = [int(n.strip()) for n in args.N_list.split(",")]
    records = []
    for N in N_values:
        print(f"[{args.model}] Running for N = {N} ...", flush=True)

        # 5a) Time the pricing call
        t0 = time.time()
        price, boundaries = price_bermudan_cos(
            S0=args.S0,
            K=args.K,
            r=args.r,
            T=args.T,
            M=args.M,
            N=N,
            a=a,
            b=b,
            charfn=charfn,           # pass the model‐specific CF
            option_type=args.option_type
        )
        t1 = time.time()
        runtime = (t1 - t0) * 1000.0   # milliseconds

        records.append({
            "model": args.model,
            "N": N,
            "price": price,
            "runtime_ms": runtime,
            "boundary_m1": boundaries[1],
            "boundary_m2": boundaries[2],
            # … you can store more of the boundary list if desired …
        })

    # 6) Build DataFrame and save
    df = pd.DataFrame(records)
    df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")

    # 7) Plot price vs N
    plt.figure(figsize=(6, 4))
    plt.plot(df["N"], df["price"], marker="o", linestyle="-")
    plt.xscale("log", base=2)
    plt.xlabel("Number of COS terms (N)")
    plt.ylabel("Option Price")
    plt.title(f"{args.model} Bermudan {args.option_type.title()} Price vs N")
    plt.grid(True, linestyle="--", alpha=0.6)
    price_plot_file = os.path.join(
        args.plots_dir,
        f"{args.model}_price_vs_N_{args.option_type}.png"
    )
    plt.savefig(price_plot_file, bbox_inches="tight")
    plt.close()
    print(f"Price plot saved to {price_plot_file}")

    # 8) Plot runtime vs N (log-log)
    plt.figure(figsize=(6, 4))
    plt.plot(df["N"], df["runtime_ms"], marker="o", linestyle="-", color="C1")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Number of COS terms (N)")
    plt.ylabel("Runtime (ms)")
    plt.title(f"{args.model} Runtime vs N (M={args.M}, L={args.L})")
    plt.grid(True, linestyle="--", alpha=0.6)
    runtime_plot_file = os.path.join(
        args.plots_dir,
        f"{args.model}_runtime_vs_N_{args.option_type}.png"
    )
    plt.savefig(runtime_plot_file, bbox_inches="tight")
    plt.close()
    print(f"Runtime plot saved to {runtime_plot_file}")


if __name__ == "__main__":
    main()
