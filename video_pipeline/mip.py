"""Bound native MIP calls in a spawned worker, preserving the feasible incumbent."""
import time
import multiprocessing as mp
import numpy as np

def native(cost, matrix, lower, upper, initial, seconds, integer_variables):
    from scipy.optimize._highspy import _core as hs
    m = matrix.tocsc()
    N = m.shape[1]
    V = integer_variables
    lp = hs.HighsLp()
    lp.num_col_ = N
    lp.num_row_ = m.shape[0]
    lp.col_cost_ = cost
    lp.col_lower_ = np.zeros(N)
    lp.col_upper_ = np.r_[np.ones(V), np.full(N - V, hs.kHighsInf)]
    lp.row_lower_ = np.where(np.isneginf(lower), -hs.kHighsInf, lower)
    lp.row_upper_ = np.where(np.isposinf(upper), hs.kHighsInf, upper)
    lp.integrality_ = [hs.HighsVarType.kInteger] * V + [hs.HighsVarType.kContinuous] * (N - V)
    lp.a_matrix_.format_ = hs.MatrixFormat.kColwise
    lp.a_matrix_.start_ = m.indptr
    lp.a_matrix_.index_ = m.indices
    lp.a_matrix_.value_ = m.data
    solver = hs._Highs()
    solver.setOptionValue('threads', 8)
    solver.setOptionValue('time_limit', max(0.01, seconds))
    solver.setOptionValue('mip_rel_gap', 0.001)
    solver.setOptionValue('output_flag', False)
    solver.passModel(lp)
    seed = hs.HighsSolution()
    seed.col_value = initial
    seed.value_valid = True
    solver.setSolution(seed)
    if solver.run() == hs.HighsStatus.kError:
        raise RuntimeError('Integer solver failed')
    result = solver.getSolution()
    return (np.asarray(result.col_value) if result.value_valid else initial, str(solver.getModelStatus()))

def worker(values, status, cost, matrix, lower, upper, initial, seconds, integer_variables):
    try:
        (answer, message) = native(cost, matrix, lower, upper, initial, seconds, integer_variables)
        np.frombuffer(values, dtype=np.float64)[:] = answer
        status.value = message.encode()[:500]
    except BaseException as e:
        status.value = ('error: ' + repr(e)).encode()[:500]

def isolated(cost, matrix, lower, upper, initial, seconds, integer_variables):
    if seconds <= 0:
        return (initial, 'wall_clock_time_limit_keep_feasible')
    ctx = mp.get_context('spawn')
    values = ctx.RawArray('d', len(initial))
    status = ctx.Array('c', 512)
    np.frombuffer(values, dtype=np.float64)[:] = initial
    process = ctx.Process(target=worker, args=(values, status, cost, matrix, lower, upper, initial, seconds, integer_variables))
    started = time.monotonic()
    process.start()
    try:
        process.join(max(0.0, seconds - (time.monotonic() - started)))
        if process.is_alive():
            process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join()
            return (initial, 'wall_clock_time_limit_keep_feasible')
        message = status.value.decode()
        if process.exitcode or message.startswith('error:') or (not message):
            raise RuntimeError('MIP worker failed: ' + message + ' exit=' + str(process.exitcode))
        return (np.frombuffer(values, dtype=np.float64).copy(), message)
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
