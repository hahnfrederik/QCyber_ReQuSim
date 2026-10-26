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


con = duckdb.connect("database/constW.db")


params = {
    "P_LINK": 0.8,
    "LENGTH": 20e3,
    "COMMUNICATION_SPEED": 2e8,
    "L_ATT": 22e3,
}
graph = GraphsReq(
    N=4,
    E=[
        (0, 1),
        (0, 2),
        (0, 3),
    ],
    sets=[[1, 2, 3], [0]],
)


white_noise_list = np.linspace(0.6, 1, num=20)
mem_noise_list = np.logspace(-3, 1, num=20, base=2)
max_iter = 5
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
    standard_graph_state_evaluation(data_frame=df, graph=graph, option="comp")
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
    standard_graph_state_evaluation(data_frame=df, graph=graph, option="comp")
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
    ],
)
results_2 = pd.DataFrame(
    data=result_list_2,
    index=index,
    columns=[
        "raw_rate",
        "fidelity",
        "fidelity_std_err",
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
# ax.plot(mem_noise_list, results_2.values[:,1]- results_1.values[0,1])
plt.savefig(
    f"../images/mem_noise_subs_sameW/mem_noise_mesh_0.8_test_2.png", bbox_inches="tight"
)
