from requsim_ent_dist_scen_1 import TCP_graph_protocol_scenario_1 as TCP1, run as run_1
from requsim_ent_dist_scen_2 import TCP_graph_protocol_scenario_2 as TCP2, run as run_2
from GraphsReq import GraphsReq
from requsim.tools.evaluation import standard_graph_state_evaluation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import LogFormatterMathtext

import duckdb

# P_LINK_LIST = np.logspace(0, -4, num=5, base=np.e)
P_LINK_LIST = [0.6, 0.7, 0.8, 0.85, 0.9]
LENGTH_LIST = [1e3, 2e3, 5e3, 10e3, 20e3, 50e3]
N_LIST = [4, 6, 8]

for g_size in N_LIST:
    for P_LINK in P_LINK_LIST:
        for LENGTH in LENGTH_LIST:
            params = {
                "P_LINK": P_LINK,
                "LENGTH": LENGTH,
                "COMMUNICATION_SPEED": 2e8,
                "L_ATT": 22e3,
            }
            graph = GraphsReq(
                N=g_size,
                E=[(0, i) for i in range(1, g_size)],
                sets=[[i for i in range(1, g_size)], [0]],
            )

            white_noise_list = np.linspace(0.6, 1, num=5)
            mem_noise_list = np.logspace(-3, 1, num=5, base=2)
            max_iter = 100
            X, Y = np.meshgrid(white_noise_list, mem_noise_list)
            raw_data_1 = [
                run_1(
                    w_noise=w_noise,
                    graph=graph,
                    max_iter=max_iter,
                    params=params,
                ).data
                for w_noise in white_noise_list
                # for mem_noise in mem_noise_list
            ]
            result_list_1 = [
                standard_graph_state_evaluation(
                    data_frame=df, graph=graph, option="comp"
                )
                for df in raw_data_1
            ]
            raw_data_2 = [
                run_2(
                    w_noise=w_noise,
                    mem_noise=mem_noise,
                    graph=graph,
                    max_iter=max_iter,
                    params=params,
                ).data
                for w_noise in white_noise_list
                for mem_noise in mem_noise_list
            ]
            result_list_2 = [
                standard_graph_state_evaluation(
                    data_frame=df, graph=graph, option="comp"
                )
                for df in raw_data_2
            ]

            index = pd.MultiIndex.from_product(
                [mem_noise_list, white_noise_list], names=["mem_noise", "white_noise"]
            )
            # index = white_noise_list

            results_1 = pd.DataFrame(
                data=result_list_1,
                index=white_noise_list,
                # index = index,
                columns=[
                    "raw_rate",
                    "fidelity",
                    "fidelity_std_err",
                    "key_per_time",
                    "key_per_time_std_err",
                    "e_x",
                    "e_z",
                    "max_vote_rate",
                ],
            )
            results_2 = pd.DataFrame(
                data=result_list_2,
                index=index,
                columns=[
                    "raw_rate",
                    "fidelity",
                    "fidelity_std_err",
                    "key_per_time",
                    "key_per_time_std_err",
                    "e_x",
                    "e_z",
                    "max_vote_rate",
                ],
            )

            fig, ax = plt.subplots()
            Z = (
                (
                    results_2.values[:, 1]
                    - (
                        results_1.values[:, 1]
                        .reshape(1, len(white_noise_list))
                        .repeat(len(mem_noise_list), axis=1)
                    )
                )
                .reshape(len(mem_noise_list), len(white_noise_list))
                .transpose()
            )

            # Z = (results_2.values[:,1]-results_1.values[:,1]).reshape(len(mem_noise_list), len(white_noise_list))
            norm = TwoSlopeNorm(vmin=np.min(Z), vcenter=0, vmax=np.max(Z))
            mesh = ax.pcolormesh(X, Y, Z, cmap="PuOr", norm=norm)
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(LogFormatterMathtext(base=2))
            # ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
            ax.set_xlabel(r"white-noise parameter $p_w$", labelpad=9)
            ax.set_ylabel(r"dephasing time $T_{\text{dp}}$ [$s$]", labelpad=10)
            cbar = fig.colorbar(mesh, ax=ax)
            cbar.set_label(r"difference of fidelity $F_{\mathrm{diff}}$", labelpad=11)
            plt.savefig(
                f"../images/diffW/diffW_size{g_size}_{P_LINK}_{LENGTH/10e3}km.png",
                bbox_inches="tight",
            )
            results_1 = results_1.reset_index().rename(columns={"index": "white_noise"})
            results_1["P_LINK"] = P_LINK
            results_1["LENGTH"] = LENGTH
            results_1["N"] = g_size
            results_2 = results_2.reset_index()

            results_2["P_LINK"] = P_LINK
            results_2["LENGTH"] = LENGTH
            results_2["N"] = g_size

            con = duckdb.connect("data/run_test.db")

            con.execute("INSERT INTO diffW_scen_1 BY NAME SELECT * from results_1")

            con.execute("INSERT INTO diffW_scen_2 BY NAME SELECT * from results_2")

            con.close()
