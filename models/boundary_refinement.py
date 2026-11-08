"""Physics-based Transition Boundary Refinement Module (Phase 5.1).
Refines macroeconomic Viterbi transition boundaries using exact kinematic signals:
1. Entering DEAD (D): ball physically crossing pitch boundary or coming to a complete halt (< 0.6 m/s).
2. Exiting DEAD (D -> H / A): first controlled contact by the restarting team (< 2.5 m).
3. Team Switch (H <-> A): exact moment spatial dominance crosses over (< 2.5 m).
Prevents overfitting: pure physical rules with zero trainable parameters.
"""

import numpy as np

# Alphabet
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
    """Refine transition boundaries using physical kinematics and ball proximity.
    
    Args:
        labels: 1D array of predicted state labels in {0: H, 1: A, 2: D, 3: X}
        d_h: Distance from ball to nearest Home player (meters)
        d_a: Distance from ball to nearest Away player (meters)
        ball_out: 1 if ball is physically outside pitch boundaries, 0 otherwise
        ball_speed: Optional ball speed array (m/s)
        window_dead: Maximum search window backwards for dead-ball transitions (frames)
        window_live: Maximum search window for possession control (frames)
        
    Returns:
        refined_labels: Refined state labels with exact kinematic boundaries
    """
    refined = labels.copy()
    N = len(refined)
    change_idx = np.where(refined[:-1] != refined[1:])[0] + 1

    for t_idx in change_idx:
        prev_lbl = refined[t_idx - 1]
        next_lbl = refined[t_idx]

        # Ignore transitions involving NOBODY (X)
        if prev_lbl == NOBODY or next_lbl == NOBODY:
            continue

        # 1. Entering DEAD (D): ball out of pitch or stopped
        if next_lbl == DEAD:
            w_start = max(0, t_idx - window_dead)
            w_end = min(N, t_idx + 10)

            # Check if ball went out of pitch in this window
            outs = np.where(ball_out[w_start:w_end] == 1)[0]
            if len(outs) > 0:
                first_out = w_start + outs[0]
                if first_out < t_idx:
                    refined[first_out:t_idx] = DEAD
                elif first_out > t_idx:
                    refined[t_idx:first_out] = prev_lbl
            elif ball_speed is not None:
                # Check where ball came to a stop (< 0.6 m/s) with no player controlling it (> 2.0 m)
                stops = np.where(
                    (ball_speed[w_start:w_end] < 0.6)
                    & (np.minimum(d_h[w_start:w_end], d_a[w_start:w_end]) > 2.0)
                )[0]
                if len(stops) > 0:
                    first_stop = w_start + stops[0]
                    if first_stop < t_idx:
                        refined[first_stop:t_idx] = DEAD

        # 2. Exiting DEAD to ALIVE (D -> H or D -> A): first contact on restart
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

        # 3. Turnover between teams (H <-> A): exact point where advantage crosses over
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
