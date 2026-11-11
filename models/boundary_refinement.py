import numpy as np

HOME, AWAY, DEAD, NOBODY = 0, 1, 2, 3

def refine_transition_boundaries(
    labels,
    d_h,
    d_a,
    ball_out,
    ball_speed=None,
    window_dead=35,
    window_live=15,
):
    refined = labels.copy()
    N = len(refined)
    change_idx = np.where(refined[:-1] != refined[1:])[0] + 1

    for t_idx in change_idx:
        prev_lbl = refined[t_idx - 1]
        next_lbl = refined[t_idx]

        if prev_lbl == NOBODY or next_lbl == NOBODY:
            continue

        if next_lbl == DEAD:
            w_start = max(0, t_idx - window_dead)
            w_end = min(N, t_idx + 10)

            outs = np.where(ball_out[w_start:w_end] == 1)[0]
            if len(outs) > 0:
                first_out = w_start + outs[0]
                if first_out < t_idx:
                    refined[first_out:t_idx] = DEAD
                elif first_out > t_idx:
                    refined[t_idx:first_out] = prev_lbl
            elif ball_speed is not None:
                stops = np.where(
                    (ball_speed[w_start:w_end] < 0.6)
                    & (np.minimum(d_h[w_start:w_end], d_a[w_start:w_end]) > 2.0)
                )[0]
                if len(stops) > 0:
                    first_stop = w_start + stops[0]
                    if first_stop < t_idx:
                        refined[first_stop:t_idx] = DEAD

        elif prev_lbl == DEAD and next_lbl in [HOME, AWAY]:
            w_start = max(0, t_idx - window_live)
            w_end = min(N, t_idx + 25)
            target_d = d_h if next_lbl == HOME else d_a

            local_min = np.argmin(target_d[w_start:w_end])
            best_f = w_start + local_min
            if target_d[best_f] < 2.5:
                if best_f < t_idx:
                    refined[best_f:t_idx] = next_lbl
                elif best_f > t_idx:
                    refined[t_idx:best_f] = DEAD

        elif prev_lbl in [HOME, AWAY] and next_lbl in [HOME, AWAY] and prev_lbl != next_lbl:
            w_start = max(0, t_idx - window_live)
            w_end = min(N, t_idx + window_live)
            target_d = d_h if next_lbl == HOME else d_a
            rival_d = d_a if next_lbl == HOME else d_h

            adv = rival_d[w_start:w_end] - target_d[w_start:w_end]
            crosses = np.where((adv > 0) & (target_d[w_start:w_end] < 2.5))[0]
            if len(crosses) > 0:
                first_cross = w_start + crosses[0]
                if first_cross < t_idx:
                    refined[first_cross:t_idx] = next_lbl
                elif first_cross > t_idx:
                    refined[t_idx:first_cross] = prev_lbl

    return refined
