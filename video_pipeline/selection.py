"""Select whole videos using grouped alternatives and ordered hour objectives."""

from dataclasses import dataclass
from itertools import product
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack

from .common import DIMENSIONS


@dataclass
class Option:
    group: int
    indices: np.ndarray
    stats: np.ndarray  # Hours, GiB, violation hours, density rank hours, quality rank hours.


def options(data, groups, config, seed, targets=None):
    ratio = data.b / data.h
    order = np.lexsort((data.rank, -ratio))
    density_rank = np.empty(data.n)
    density_rank[order] = (np.arange(data.n) + 1) / max(data.n, 1)
    features = np.column_stack([data.h, data.b, data.h * data.penalty,
                                data.h * density_rank, data.h * data.rank])
    result, empty_choices, seed_choices = [], [], []
    limit = config["max_options_per_group"]
    for group, (key, indices) in enumerate(groups):
        seen = {}

        def add(selected, stats=None):
            amounts = features[selected].sum(axis=0) if stats is None else stats
            signature = (len(selected), *np.round(amounts, 10))
            if signature in seen:
                return seen[signature]
            if len(indices) > 5 and len(seen) >= limit:
                return None
            number = len(result)
            seen[signature] = number
            result.append(Option(group, selected, amounts))
            return number

        empty_choices.append(add(indices[:0]))
        seed_choices.append(add(indices[seed[indices]]))
        add(indices)
        if len(indices) <= 5:
            for mask in product([False, True], repeat=len(indices)):
                add(indices[np.asarray(mask)])
            continue
        sequences = [indices[np.lexsort((data.rank[indices], -ratio[indices]))],
                     indices[np.lexsort((data.rank[indices], ratio[indices]))],
                     indices[np.argsort(data.rank[indices], kind="stable")]]
        total = config["target_total_hours"]
        target = total * data.h[indices].sum() / data.h.sum() if targets is None else targets[group]
        hints = [target, total]
        for position, dimension in enumerate(DIMENSIONS):
            for rule in config["rules"][dimension]:
                if key[position] == rule["id"]:
                    hints.extend(total * rule[bound] for bound in ["min_share", "max_share"]
                                 if rule.get(bound) is not None)
        for sequence in sequences:
            cumulative = np.vstack([np.zeros(5), np.cumsum(features[sequence], axis=0)])
            cuts = []
            for hours in hints + [cumulative[-1, 0] * fraction for fraction in [.25, .5, .75]]:
                index = min(len(sequence), int(np.searchsorted(cumulative[:, 0], hours)))
                for cut in [index, max(0, index - 1), min(len(sequence), index + 1)]:
                    if cut not in cuts:
                        cuts.append(cut)
            count = len(seen)
            for cut in cuts:
                add(sequence[:cut], cumulative[cut])
                if len(seen) - count >= max(1, (limit - 3) // len(sequences)):
                    break
    return result, np.asarray(empty_choices), np.asarray(seed_choices)


def build(data, groups, choices, config):
    count, group_count = len(choices), len(groups)
    stats = np.asarray([choice.stats for choice in choices])
    membership = np.asarray([choice.group for choice in choices])
    hours, gib = stats[:, 0], stats[:, 1]
    target, cap = config["target_total_hours"], config["max_gib_per_hour"]
    tolerance = config["total_tolerance"]
    equality = coo_matrix((np.ones(count), (membership, np.arange(count))),
                          shape=(group_count, count)).tocsr()
    hard_rows = [csr_matrix((gib - cap * hours).reshape(1, -1))]
    soft_rows, soft_bounds, priorities = [], [], []

    def soft(values, lower, upper, priority):
        soft_rows.append(csr_matrix(values.reshape(1, -1)))
        soft_bounds.append((lower, upper))
        priorities.append(priority)

    soft(hours, target * (1 - tolerance), target * (1 + tolerance), 0)
    for position, dimension in enumerate(DIMENSIONS):
        labels = np.asarray([key[position] for key, _ in groups], dtype=object)[membership]
        if dimension != "language":
            for label in dict.fromkeys(labels):
                if label is not None:
                    hard_rows.append(csr_matrix(((gib - cap * hours) * (labels == label)).reshape(1, -1)))
        for rule in config["rules"][dimension]:
            upper = rule.get("max_share")
            soft(hours * (labels == rule["id"]), target * rule.get("min_share", 0),
                 target * upper if upper is not None else np.inf, 1)
    thresholds = sorted(rule["min_short_side"] for rule in config["technical_quality"]["resolution_shares"])
    bands = np.asarray([key[4] for key, _ in groups])[membership]
    for rule in config["technical_quality"]["resolution_shares"]:
        hit = bands >= thresholds.index(rule["min_short_side"]) + 1
        soft(hours * hit, target * rule["min_share"], np.inf, 3)
    hard = vstack([equality, *hard_rows], format="csr")
    hard_lower = np.r_[np.ones(group_count), np.full(len(hard_rows), -np.inf)]
    hard_upper = np.r_[np.ones(group_count), np.zeros(len(hard_rows))]
    rows, signs, bounds, slack_priorities = [], [], [], []
    for row, (lower, upper), priority in zip(soft_rows, soft_bounds, priorities):
        if np.isfinite(lower):
            rows.append(row); signs.append(1.); bounds.append((lower, np.inf)); slack_priorities.append(priority)
        if np.isfinite(upper):
            rows.append(row); signs.append(-1.); bounds.append((-np.inf, upper)); slack_priorities.append(priority)
    size = len(rows)
    soft_matrix = vstack(rows, format="csr")
    slack = coo_matrix((signs, (np.arange(size), np.arange(size))), shape=(size, size)).tocsr()
    matrix = vstack([hstack([hard, csr_matrix((hard.shape[0], size))]),
                     hstack([soft_matrix, slack])], format="csr")
    lower = np.r_[hard_lower, [bound[0] for bound in bounds]]
    upper = np.r_[hard_upper, [bound[1] for bound in bounds]]
    slack_priorities = np.asarray(slack_priorities)
    objectives = [
        ("total_hour_distance", np.r_[np.zeros(count), (slack_priorities == 0).astype(float)]),
        ("category_hour_distance", np.r_[np.zeros(count), (slack_priorities == 1).astype(float)]),
        ("technical_violation_hours", np.r_[stats[:, 2], np.zeros(size)]),
        ("resolution_hour_shortfall", np.r_[np.zeros(count), (slack_priorities == 3).astype(float)]),
        ("density_rank_hours", np.r_[stats[:, 3], np.zeros(size)]),
        ("quality_rank_hours", np.r_[stats[:, 4], np.zeros(size)]),
    ]

    def complete(values):
        actual = np.asarray(soft_matrix @ values).ravel()
        distance = [max(0, low - value) if sign == 1 else max(0, value - high)
                    for value, (low, high), sign in zip(actual, bounds, signs)]
        return np.r_[values, distance]

    return matrix, lower, upper, objectives, complete, hard, hard_lower, hard_upper, stats


def feasible(values, matrix, lower, upper, variable_upper):
    actual = np.asarray(matrix @ values).ravel()
    return (np.isfinite(values).all() and np.all(values >= -1e-7) and
            np.all(values <= variable_upper + 1e-7) and
            np.all(actual >= lower - 1e-7) and np.all(actual <= upper + 1e-7))


def continuous(model, seconds, initial):
    matrix, lower, upper, objectives, complete, hard, *_ = model
    count = hard.shape[1]
    limits = np.r_[np.ones(count), np.full(matrix.shape[1] - count, np.inf)]
    current = complete(initial)
    phases = []
    for name, cost in objectives:
        high, low = np.isfinite(upper), np.isfinite(lower)
        result = linprog(cost, A_ub=vstack([matrix[high], -matrix[low]], format="csr"),
                         b_ub=np.r_[upper[high], -lower[low]], bounds=list(zip(np.zeros(len(limits)), limits)),
                         method="highs", options={"time_limit": seconds, "threads": 8,
                                                  "primal_feasibility_tolerance": 1e-9,
                                                  "dual_feasibility_tolerance": 1e-9})
        phases.append(dict(phase=name, status=int(result.status)))
        if not result.success:
            if result.status != 1:
                raise RuntimeError(f"Continuous solver failed at {name}: " + result.message)
            if result.x is not None and feasible(result.x, matrix, lower, upper, limits) and cost @ result.x <= cost @ current + 1e-7:
                current = result.x
            break
        current = result.x
        matrix = vstack([matrix, csr_matrix(cost.reshape(1, -1))], format="csr")
        lower = np.r_[lower, -np.inf]
        upper = np.r_[upper, float(cost @ current) + 1e-7]
    return current[:count], phases


def integer(model, initial, seconds):
    from .mip import isolated, native
    matrix, lower, upper, objectives, complete, hard, *_ = model
    count = hard.shape[1]
    current = complete(initial)
    limits = np.r_[np.ones(count), np.full(len(current) - count, np.inf)]
    phases = []
    deadline = time.monotonic() + seconds
    for step, (name, cost) in enumerate(objectives):
        if float(cost @ current) <= 1e-10:
            status = "zero_distance"
        elif time.monotonic() >= deadline:
            status = "time_limit_keep_feasible"
        else:
            runner = isolated if count >= 10000 else native
            raw, status = runner(cost, matrix, lower, upper, current,
                                 (deadline - time.monotonic()) / (len(objectives) - step), count)
            trial = complete(np.rint(raw[:count]))
            if feasible(trial, matrix, lower, upper, limits) and cost @ trial <= cost @ current + 1e-7:
                current = trial
        objective = float(cost @ current)
        phases.append(dict(phase=name, status=status, objective=objective))
        matrix = vstack([matrix, csr_matrix(cost.reshape(1, -1))], format="csr")
        lower = np.r_[lower, -np.inf]
        upper = np.r_[upper, objective + 1e-7]
    return current[:count], phases


def select(data, config, seed):
    if data.n == 0:
        return np.empty(0, dtype=np.int64), dict(groups=0, options=0, phases=[], lp_phases=[])
    groups, targets, lp_phases = data.groups(), None, []
    for round_number in range(config["refinement_rounds"]):
        choices, empty, seeded = options(data, groups, config, seed, targets)
        model = build(data, groups, choices, config)
        hard, lower, upper = model[5:8]
        warm = np.zeros(len(choices)); warm[seeded] = 1
        if not feasible(warm, hard, lower, upper, np.ones(len(warm))):
            warm[:] = 0; warm[empty] = 1
        fraction, phases = continuous(model, config["lp_seconds"], warm)
        lp_phases.append(dict(round=round_number + 1, phases=phases))
        targets = np.zeros(len(groups))
        for index, choice in enumerate(choices):
            targets[choice.group] += fraction[index] * choice.stats[0]
    selected, phases = integer(model, warm, config["solve_seconds"])
    indices = np.sort(np.concatenate([choice.indices for index, choice in enumerate(choices) if selected[index] > .5]))
    return indices, dict(groups=len(groups), options=len(choices), phases=phases, lp_phases=lp_phases)
