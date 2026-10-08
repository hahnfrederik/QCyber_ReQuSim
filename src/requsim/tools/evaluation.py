import os
import numpy as np
from ..libs import matrix as mat
import pandas as pd
from warnings import warn
from scipy import linalg


def binary_entropy(p):
    """Calculate the binary entropy.

    Parameters
    ----------
    p : scalar
        Must be in interval [0, 1]. Usually an error rate.

    Returns
    -------
    scalar
        The binary entropy of `p`.

    """
    if p == 1 or p == 0:
        return 0
    else:
        res = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        if np.isnan(res):
            warn(f"binary_entropy was called with p={p} and returned nan")
        return res


def calculate_keyrate_time(
    correlations_z, correlations_x, err_corr_ineff, time_interval, return_std_err=False
):
    """Calculate the asymptotic key rate per time from a list of correlations.

    This uses the sample mean of error rates to estimate what the asymptotic
    key rate would be. It uses the formula for a bound on the asymptotic key
    rate. See for this particular formulation:

    D. Luong, L. Jiang, J. Kim, N. Lütkenhaus; Appl. Phys. B 122, 96 (2016)
    arXiv:1508.02811 [quant-ph]

    Optionally also returns the standard deviation of the key rate
    calculated by using propagation of error.

    Parameters
    ----------
    correlations_z : list of scalars
        List of correlations. The entries should correspond to the probabilty
        that measurement outcomes in z-direction coincide for each raw bit.
    correlations_x : list of scalars
        List of correlations. The entries should correspond to the probabilty
        that measurement outcomes in x-direction coincide for each raw bit.
    err_corr_ineff : scalar
        The error correction inefficiency, which lowers the obtainable key rate.
        1 means perfectly efficient; >1 indicates inefficiencies
    time_interval : scalar
        Time interval in which the raw bits were collected.
    return_std_err : bool
        Whether to also compute and return the standard error of the mean of the
        key rate. Default: False

    Returns
    -------
    scalar or tuple of scalars
        If return_std_err is False, returns the key rate.
        If return_std_err is True, returns at tuple of key rate and
        standard error of the mean of the key rate.

    """
    e_z = 1 - np.mean(correlations_z)
    e_x = 1 - np.mean(correlations_x)
    num_pairs = len(correlations_z)
    pair_per_time = num_pairs / time_interval
    keyrate = pair_per_time * (
        1 - binary_entropy(e_x) - err_corr_ineff * binary_entropy(e_z)
    )
    if not return_std_err:
        return keyrate
    # use error propagation formula
    if e_z == 0:
        keyrate_std = pair_per_time * np.sqrt(
            (-np.log2(e_x) + np.log2(1 - e_x)) ** 2 * np.std(correlations_x) ** 2
        )
    else:
        keyrate_std = pair_per_time * np.sqrt(
            (-np.log2(e_x) + np.log2(1 - e_x)) ** 2 * np.std(correlations_x) ** 2
            + err_corr_ineff**2
            * (-np.log2(e_z) + np.log2(1 - e_z)) ** 2
            * np.std(correlations_z) ** 2
        )
    keyrate_std_err = keyrate_std / np.sqrt(num_pairs)
    return keyrate, keyrate_std_err


def calculate_keyrate_channel_use(
    correlations_z, correlations_x, err_corr_ineff, resource_list, return_std_err=False
):
    """Calculate the asymptotic key rate per resource from a list of correlations.

    CAREFUL: This formulation only makes sense when the amount of resources
    (usually number of channel uses or similar) is directly assignable to one
    particular pair or set of raw bits. This is often not the case e.g. if
    connections are not established sequentially for each bit, as would be the
    case for multi-mode memories.
    This function uses the sample mean of error rates to estimate what the
    asymptotic key rate would be.
    It uses the formula for a bound on the asymptotic key
    rate. See for this particular formulation:

    D. Luong, L. Jiang, J. Kim, N. Lütkenhaus; Appl. Phys. B 122, 96 (2016)
    arXiv:1508.02811 [quant-ph]

    Optionally also returns the standard deviation of the key rate
    calculated by using propagation of error.

    Parameters
    ----------
    correlations_z : list of scalars
        List of correlations. The entries should correspond to the probabilty
        that measurement outcomes in z-direction coincide for each raw bit.
    correlations_x : list of scalars
        List of correlations. The entries should correspond to the probabilty
        that measurement outcomes in x-direction coincide for each raw bit.
    err_corr_ineff : scalar
        The error correction inefficiency, which lowers the obtainable key rate.
        1 means perfectly efficient; >1 indicates inefficiencies
    resource_list : list of scalar
        A list containing the number of resources each set of raw bits consumed.
        This might not make sense if the number of consumed resources is not
        directly assignable to one particular set of raw bits.
    return_std_err : bool
        Whether to also compute and return the standard error of the mean of the
        key rate. Default: False

    Returns
    -------
    scalar or tuple of scalars
        If return_std_err is False, returns the key rate.
        If return_std is True, returns at tuple of key rate and
        standard deviation of the key rate.

    """
    e_z = 1 - np.mean(correlations_z)
    e_x = 1 - np.mean(correlations_x)
    num_pairs = len(correlations_z)
    pair_per_resource = num_pairs / np.sum(resource_list)
    keyrate = pair_per_resource * (
        1 - binary_entropy(e_x) - err_corr_ineff * binary_entropy(e_z)
    )
    if not return_std_err:
        return keyrate
    # use error propagation formula
    if e_z == 0:
        keyrate_std = pair_per_resource * np.sqrt(
            (-np.log2(e_x) + np.log2(1 - e_x)) ** 2 * np.std(correlations_x) ** 2
        )
    else:
        keyrate_std = pair_per_resource * np.sqrt(
            (-np.log2(e_x) + np.log2(1 - e_x)) ** 2 * np.std(correlations_x) ** 2
            + err_corr_ineff**2
            * (-np.log2(e_z) + np.log2(1 - e_z)) ** 2
            * np.std(correlations_z) ** 2
        )
    keyrate_std_err = keyrate_std / np.sqrt(num_pairs)
    return keyrate, keyrate_std_err


def calculate_keyrate_channel_use_from_time(
    correlations_z,
    correlations_x,
    err_corr_ineff,
    time_list,
    trial_time,
    return_std_err=False,
):
    """Calculate the asymptotic key rate per resource with only timing info.

    In some setups the number of channel uses is directly tied to time.
    CARFUL: This measure only makes sense if this is the case for the setup
    you are analyzing.

    This function calculates time intervals and resources from the `time_list`
    and then passes them to calculate_keyrate_channel_use.

    Parameters
    ----------
    correlations_z : list of scalars
        List of correlations. The entries should correspond to the probabilty
        that measurement outcomes in z-direction coincide for each raw bit.
    correlations_x : list of scalars
        List of correlations. The entries should correspond to the probabilty
        that measurement outcomes in x-direction coincide for each raw bit.
    err_corr_ineff : scalar
        The error correction inefficiency, which lowers the obtainable key rate.
        1 means perfectly efficient; >1 indicates inefficiencies
    time_list : list of scalar
        A list containing the point in time that a each set of raw bits was
        recorded. e.g. data["time"] from the data attribute of a
        requsim.tools.TwoLinkProtocol
        CAREFUL: This assumes that the protocol was started at time 0 and that
        the communication time is half the trial time.
    trial_time : scalar
        The time one trial to establish a pair takes. Usually something like
        preparation time + 2 * distance / communication speed.
    return_std_err : bool
        Whether to also compute and return the standard error of the mean of the
        key rate. Default: False

    Returns
    -------
    scalar or tuple of scalars
        If return_std is False, returns the key rate.
        If return_std is True, returns at tuple of key rate and
        standard deviation of the key rate.

    """
    time_interval_list = np.diff(pd.concat([pd.Series([trial_time / 2]), time_list]))
    resource_list = time_interval_list / trial_time
    return calculate_keyrate_channel_use(
        correlations_z=correlations_z,
        correlations_x=correlations_x,
        err_corr_ineff=err_corr_ineff,
        resource_list=resource_list,
        return_std_err=return_std_err,
    )


def standard_bipartite_evaluation(data_frame, err_corr_ineff=1):
    """Calculate fidelities and key rates from times and states.

    Parameters
    ----------
    data_frame : pd.DataFrame
        A pandas DataFrame with columns "time" and "state", representing
        when each connection was made and the two-qubit state associated with
        that connection.
    err_corr_ineff : scalar
        The error correction inefficiency, which lowers the obtainable key rate.
        1 means perfectly efficient; >1 indicates inefficiencies. Default: 1

    Returns
    -------
    list of scalars
        contains: raw rate,
                  average fidelity,
                  standard error of the mean of fidelity,
                  average asymptotic key rate per time,
                  standard error of the mean of key rate per time

    """
    raw_rate = len(data_frame["time"]) / data_frame["time"].iloc[-1]

    states = data_frame["state"]

    fidelity_list = np.real_if_close(
        [
            np.dot(np.dot(mat.H(mat.phiplus), state), mat.phiplus)[0, 0]
            for state in states
        ]
    )
    fidelity = np.mean(fidelity_list)
    fidelity_std_err = np.std(fidelity_list) / np.sqrt(len(fidelity_list))

    z0z0 = mat.tensor(mat.z0, mat.z0)
    z1z1 = mat.tensor(mat.z1, mat.z1)
    correlations_z = np.real_if_close(
        [
            np.dot(np.dot(mat.H(z0z0), state), z0z0)[0, 0]
            + np.dot(np.dot(mat.H(z1z1), state), z1z1)[0, 0]
            for state in states
        ]
    )
    correlations_z[correlations_z > 1] = 1

    x0x0 = mat.tensor(mat.x0, mat.x0)
    x1x1 = mat.tensor(mat.x1, mat.x1)
    correlations_x = np.real_if_close(
        [
            np.dot(np.dot(mat.H(x0x0), state), x0x0)[0, 0]
            + np.dot(np.dot(mat.H(x1x1), state), x1x1)[0, 0]
            for state in states
        ]
    )
    correlations_x[correlations_x > 1] = 1

    key_per_time, key_per_time_std_err = calculate_keyrate_time(
        correlations_z=correlations_z,
        correlations_x=correlations_x,
        err_corr_ineff=err_corr_ineff,
        time_interval=data_frame["time"].iloc[-1],
        return_std_err=True,
    )
    return [
        raw_rate,
        fidelity,
        fidelity_std_err,
        key_per_time,
        key_per_time_std_err,
    ]


def ghz_fidelities(data: pd.DataFrame, N):
    """function to calculate the fidelity of a given density matrix to the ghz state

    Parameters
    ----------
    rho: np.ndarray
        the density matrix of the quantum state that is to be compared to the ghz state
    N: integer
        the number of qubits in the system described by rho
        Theoretically, it is not needed here since this can be calculated by rho itself

    Returns
    -------
    fidelity: scalar
        the fidelity of the state rho and a ghz state
    """

    # Generalize the function for general fidelity function for two states
    z0s = [mat.z0] * N
    z0s = mat.tensor(*z0s)
    z1s = [mat.z1] * N
    z1s = mat.tensor(*z1s)
    ghz_psi = 1 / np.sqrt(2) * (z0s + z1s)

    states = data["state"]
    fidelities_list = np.real_if_close(
        [np.dot(np.dot(mat.H(ghz_psi), state), ghz_psi)[0, 0] for state in states]
    )

    fidelity = np.mean(fidelities_list)

    fidelity_std_err = np.std(fidelities_list) / np.sqrt(len(fidelities_list))
    return fidelity, fidelity_std_err


def standard_ghz_evaluation(data_frame, N=None, err_corr_ineff=1):
    """Calculate fidelity and speed of GHZ state distribution.

    Parameters
    ----------

    data_frame : pd.DataFrame
        A pandas DataFrame with columns "time" and "state", representing
        when each connection was made and the multi_qubit ghz-state associated with that connection.
    N : int or None
        The number of parties sharing the GHZ state. If None, N will be evaluated from the data. Default: None
    err_corr_ineff : scalar
        The error correction inefficency, whihc lowers the obtainable key rate.
        1 means perfectly efficient; >1 indicated inefficiencies. Default: 1

    Returns
    -------

    list of scalars
       contains: raw rate,
                 average fidelity,
                 standard error of the mean fidelity,
                 average aymptotic key rate per time,
                 standard error of the mean of key rate per time
    """

    if N is None:
        N = int(np.log2(data_frame["state"][0].shape[0]))

    raw_rate = len(data_frame["time"]) / data_frame["time"].iloc[-1]
    states = data_frame["state"]
    fidelity, fidelity_std_err = ghz_fidelities(data=data_frame, N=N)

    return [
        raw_rate,
        fidelity,
        fidelity_std_err,
    ]


def ggraph_state_fidelity(data: pd.DataFrame, graph):
    """Calculate the statistical fidelity of the raph states"""
    states = data["state"]

    """only necessary if both rho and sigma are not pure
    sqrt_graph = linalg.sqrtm(graph.rho_graph)
    sqrt_graph = np.real_if_close(sqrt_graph)
    """
    fidelities = np.real_if_close(
        [mat.H(graph.psi) @ state @ graph.psi for state in states]
    )
    fidelity = np.mean(fidelities)
    fidelity_std_err = np.std(fidelities) / np.sqrt(len(fidelities))
    return fidelity, fidelity_std_err


def standard_graph_state_evaluation(data_frame, graph, err_corr_ineff=1):
    """Calculate fidelities and key rates from times and states.

    Parameters
    ----------
    data_frame : pd.DataFrame
        A pandas DataFrame with columns "time" and "state", representing
        when each connection was made and the two-qubit state associated with
        that connection.
    err_corr_ineff : scalar
        The error correction inefficiency, which lowers the obtainable key rate.
        1 means perfectly efficient; >1 indicates inefficiencies. Default: 1

    Returns
    -------
    list of scalars
        contains: raw rate,
                  average fidelity,
                  standard error of the mean of fidelity,
                  average asymptotic key rate per time,
                  standard error of the mean of key rate per time

    """

    raw_rate = len(data_frame["time"]) / data_frame["time"].iloc[-1]

    fidelity, fidelity_std_err = ggraph_state_fidelity(data=data_frame, graph=graph)

    return [raw_rate, fidelity, fidelity_std_err]
