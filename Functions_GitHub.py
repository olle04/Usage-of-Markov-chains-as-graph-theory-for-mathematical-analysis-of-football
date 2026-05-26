# All of the functions ffrom chapter 7 in networks
import numpy as np
import pandas as pd

def adjacency_matrix(df, team_id):
    is_team = df["team_id"] == team_id
    is_pass = df["type"] == "pass"
    is_accurate = df["pass_accurate"] == True
    is_in_play = (
        (df["throw_in"] == False) &
        (df["corner"] == False) &
        (df["goal_kick"] == False) &
        (df["free_kick"] == False)
    )
    passes = df.loc[is_team & is_pass & is_accurate & is_in_play,
                    ["player_id", "pass_recipient_id"]].copy()

    passes = passes.dropna(subset=["player_id", "pass_recipient_id"])

    A_id = pd.crosstab(passes["pass_recipient_id"], passes["player_id"])

    players = pd.Index(sorted(set(passes["player_id"]).union(set(passes["pass_recipient_id"]))))
    A_id = A_id.reindex(index=players, columns=players, fill_value=0).astype(int)

    id_to_name = (
        df[["player_id", "player_name"]]
        .dropna()
        .drop_duplicates("player_id")
        .set_index("player_id")["player_name"]
    )

    A_name = A_id.rename(index=id_to_name, columns=id_to_name)

    A = A_id.to_numpy()

    col_sums = A_id.sum(axis=0)
    P_id = A_id.divide(col_sums.replace(0, 1), axis=1)
    P = P_id.to_numpy()

    return A, P, A_id, A_name, id_to_name

def most_probable_path_matrix(P):
    P = np.asarray(P, dtype=float)
    n = P.shape[0]

    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0.0)

    for s in range(n):
        dist = np.full(n, np.inf)
        visited = np.zeros(n, dtype=bool)
        dist[s] = 0.0

        for _ in range(n):
            u = -1
            best = np.inf

            for i in range(n):
                if not visited[i] and dist[i] < best:
                    best = dist[i]
                    u = i

            if u == -1:
                break

            visited[u] = True

            for v in range(n):
                if P[v, u] > 0 and not visited[v]:
                    length = -np.log(P[v, u])
                    new_dist = dist[u] + length

                    if new_dist < dist[v]:
                        dist[v] = new_dist

        D[s] = dist

    return D

def signed_balance_matrix(df_stories, idx_to_player_id, id_to_name):
    mask = (
        (df_stories["team_name"] == "Hammarby") &
        (df_stories["type"] == "pass") &
        (df_stories["throw_in"] == False) &
        (df_stories["corner"] == False) &
        (df_stories["goal_kick"] == False) &
        (df_stories["free_kick"] == False) &
        (df_stories["player_id"] > 0) &
        (df_stories["pass_recipient_id"] > 0)
    )

    df_pass = df_stories.loc[
        mask,
        ["player_id", "pass_recipient_id", "start_x", "end_x"]
    ].copy()

    df_pass["start_x"] *= 1.2
    df_pass["end_x"]   *= 1.2

    n = len(idx_to_player_id)
    player_to_idx = {pid: i for i, pid in enumerate(idx_to_player_id)}

    forward_counts = np.zeros((n, n), dtype=int)
    backward_counts = np.zeros((n, n), dtype=int)

    for _, row in df_pass.iterrows():
        pid = row["player_id"]
        rid = row["pass_recipient_id"]

        if pid not in player_to_idx or rid not in player_to_idx:
            continue

        i = player_to_idx[pid]
        j = player_to_idx[rid]

        a, b = sorted([i, j])

        dx = row["end_x"] - row["start_x"]

        if dx > 0:
            forward_counts[a, b] += 1
        elif dx < 0:
            backward_counts[a, b] += 1
        # if dx == 0: ignore

    S = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(i + 1, n):
            f = forward_counts[i, j]
            b = backward_counts[i, j]

            if f > b:
                S[i, j] = 1
                S[j, i] = 1
            elif b > f:
                S[i, j] = -1
                S[j, i] = -1
            else:
                S[i, j] = 0
                S[j, i] = 0

    player_names = [id_to_name.get(pid, None) for pid in idx_to_player_id]
    S_df = pd.DataFrame(S, index=player_names, columns=player_names)

    return S, S_df

def largest_eigenvalue(A, max_iter = 1000, tol = 1e-8):
    A = np.asanyarray(A, dtype= float)
    n = A.shape[0]
    x = np.ones(n)

    for i in range(max_iter):
        w = A @ x
        x_old = x
        w_norm = np.linalg.norm(w)
        x_new = w / w_norm
        if np.linalg.norm(x_new - x_old) < tol:
            x = x_new
            break
        x = x_new
    return float((x @ (A @ x))/(x @ x))

def eigenvector_centrality_comp(A):
    A = np.asarray(A, dtype= float)
    eigenvals, eigenvect = np.linalg.eig(A)
    k = np.argmax(eigenvals)
    x_c = eigenvect[:,k]
    return x_c

def eigenvector_centrality_iter(A, max_iter = 1000, tol = 1e-8):
    A = np.asarray(A, dtype= float)
    n = A.shape[0]
    x = np.ones(n)
    x /= np.linalg.norm(x)

    for i in range(max_iter):
        x_new = A @ x
        x_new /= np.linalg.norm(x_new)
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x

def katz_centrality_comp(A, beta = 1):
    A = np.asarray(A, dtype= float)
    n = A.shape[0]
    I = np.eye(n)
    b = np.full(n, beta, dtype= float)
    eigenvals, eigenvect = np.linalg.eig(A)
    k = np.argmax(eigenvals)
    lambda_1 = eigenvals[k]
    alpha = lambda_1 - (lambda_1 / 10000)
    x = np.linalg.solve(I - alpha * A, b)

def katz_centrality_iter(A, beta = 1.0, max_iter = 1000, tol = 1e-10):
    A = np.asarray(A, dtype= float)
    n = A.shape[0]
    ones = np.ones(n, dtype= float)
    x = np.zeros(n, dtype= float)
    alpha = 0.9/abs(largest_eigenvalue(A))
    for i in range(max_iter):
        x_new = alpha * (A @ x) + beta * ones
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    return x

def page_rank_comp(P, alpha=0.85, beta=1.0):
    P = np.asarray(P, dtype=float)
    n = P.shape[0]

    B = np.eye(n) - alpha * P
    b = beta * np.ones(n)

    x = np.linalg.solve(B, b)
    return x

def page_rank_iter(P, alpha=0.85, beta=1.0, max_iter=1000, tol=1e-10):
    P = np.asarray(P, dtype=float)
    n = P.shape[0]

    x = np.ones(n) / n
    ones = np.ones(n)

    for _ in range(max_iter):
        x_new = alpha * (P @ x) + beta * ones

        if np.linalg.norm(x_new - x, 1) < tol:
            x = x_new
            break

        x = x_new

    return x

def authorities_hubs_comp(A):
    A = np.asarray(A, dtype=float)

    eigenvals, eigenvecs = np.linalg.eigh(A @ A.T)
    k = np.argmax(eigenvals)

    lambda1 = eigenvals[k]
    x = eigenvecs[:, k]

    if x[np.argmax(np.abs(x))] < 0:
        x = -x

    if np.linalg.norm(x) > 0:
        x = x / np.linalg.norm(x)

    y = (A.T @ x) / np.sqrt(lambda1)

    if np.linalg.norm(y) > 0:
        y = y / np.linalg.norm(y)

    if y[np.argmax(np.abs(y))] < 0:
        y = -y

    return x, y

def authorities_hubs_iter(A, max_iter=1000, tol=1e-8):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]

    x = np.ones(n)
    x = x / np.linalg.norm(x)

    lambda_old = 0.0

    for _ in range(max_iter):
        x_new = (A @ A.T) @ x

        norm = np.linalg.norm(x_new)
        if norm == 0:
            break
        x_new = x_new / norm

        lambda_new = float(x_new @ ((A @ A.T) @ x_new))

        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            lambda_old = lambda_new
            break

        x = x_new
        lambda_old = lambda_new

    if lambda_old > 0:
        y = (A.T @ x) / np.sqrt(lambda_old)
    else:
        y = np.zeros(n)

    if np.linalg.norm(x) > 0:
        x = x / np.linalg.norm(x)
    if np.linalg.norm(y) > 0:
        y = y / np.linalg.norm(y)

    if x[np.argmax(np.abs(x))] < 0:
        x = -x
    if np.linalg.norm(y) > 0 and y[np.argmax(np.abs(y))] < 0:
        y = -y

    return x, y

def closeness_centrality(D):
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    c = np.zeros(n)

    for i in range(n):
        reachable = np.isfinite(D[i])
        reachable[i] = False

        dist_sum = np.sum(D[i, reachable])

        if dist_sum > 0:
            c[i] = np.sum(reachable) / dist_sum
        else:
            c[i] = 0.0

    return c

def betweenness_centrality(A):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    c = np.zeros(n, dtype=float)

    for s in range(n):
        S = []
        P = [[] for _ in range(n)]

        sigma = np.zeros(n, dtype=float)
        sigma[s] = 1.0

        dist = np.full(n, np.inf)
        dist[s] = 0.0

        visited = np.zeros(n, dtype=bool)

        for _ in range(n):
            v = -1
            best = np.inf

            for i in range(n):
                if not visited[i] and dist[i] < best:
                    best = dist[i]
                    v = i

            if v == -1:
                break

            visited[v] = True
            S.append(v)

            for w in range(n):
                if A[w, v] > 0:
                    length = 1.0 / A[w, v]
                    new_dist = dist[v] + length

                    if new_dist < dist[w]:
                        dist[w] = new_dist
                        sigma[w] = sigma[v]
                        P[w] = [v]

                    elif np.isclose(new_dist, dist[w]):
                        sigma[w] += sigma[v]
                        P[w].append(v)

        delta = np.zeros(n, dtype=float)

        while S:
            w = S.pop()

            for v in P[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])

            if w != s:
                c[w] += delta[w]

    c = c / (n * (n - 1))

    return c

def k_core(A, k_in=1, k_out=1):
    A = np.asarray(A, dtype=float)

    B = (A > 0).astype(int)

    n = B.shape[0]
    keep = np.ones(n, dtype=bool)

    changed = True
    while changed:
        changed = False

        for i in range(n):
            if not keep[i]:
                continue

            # with your convention:
            # row i  = incoming edges to i
            # col i  = outgoing edges from i
            in_deg = np.sum(B[i, keep])
            out_deg = np.sum(B[keep, i])

            if in_deg < k_in or out_deg < k_out:
                keep[i] = False
                changed = True

    core_nodes = np.where(keep)[0].tolist()
    return keep, core_nodes

def local_clustering(A, min_passes=1):
    A = np.asarray(A, dtype=float)

    # symmetrize and threshold
    B = ((A >= min_passes) | (A.T >= min_passes)).astype(int)
    np.fill_diagonal(B, 0)

    n = B.shape[0]
    C = np.zeros(n, dtype=float)

    for i in range(n):
        neighbors = np.where(B[i] > 0)[0]
        k = len(neighbors)

        if k < 2:
            C[i] = 0.0
            continue

        connected_pairs = 0
        total_pairs = k * (k - 1) / 2

        for a in range(k):
            for b in range(a + 1, k):
                u = neighbors[a]
                v = neighbors[b]

                if B[u, v] == 1:
                    connected_pairs += 1

        C[i] = connected_pairs / total_pairs

    return C

def reciprocity(A):
    A = np.asarray(A, dtype=float)
    B = (A > 0).astype(int)
    np.fill_diagonal(B, 0)

    m = np.sum(B)

    if m == 0:
        return 0.0

    r = np.trace(B @ B) / m
    return float(r)

def reciprocity_local(A):
    A = np.asarray(A, dtype=float)
    B = (A > 0).astype(int)
    np.fill_diagonal(B, 0)

    n = B.shape[0]
    r = np.zeros(n, dtype=float)

    for i in range(n):
        # with your convention:
        # row i    = incoming edges to i
        # column i = outgoing edges from i
        k_in = np.sum(B[i, :])
        k_out = np.sum(B[:, i])

        mutual_neighbors = np.sum(B[i, :] * B[:, i])

        denom = k_in + k_out

        if denom > 0:
            r[i] = 2.0 * mutual_neighbors / denom
        else:
            r[i] = 0.0

    return r

def reciprocity_pairwise(P, eps=None):
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    R = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                R[i, j] = 0.0
                continue

            p_ij = P[i, j]
            p_ji = P[j, i]

            if eps is not None:
                R[i, j] = abs(np.log(p_ij + eps) - np.log(p_ji + eps))
            else:
                if p_ij == 0 and p_ji == 0:
                    R[i, j] = 0.0
                elif p_ij == 0 or p_ji == 0:
                    R[i, j] = np.inf
                else:
                    R[i, j] = abs(np.log(p_ij) - np.log(p_ji))

    return R

def structural_balance_global(S):
    S = np.asarray(S, dtype=int)
    n = S.shape[0]

    balanced = 0
    total = 0

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if S[i, j] != 0 and S[j, k] != 0 and S[k, i] != 0:
                    total += 1

                    if S[i, j] * S[j, k] * S[k, i] > 0:
                        balanced += 1

    if total == 0:
        return 0.0

    return balanced / total

def structural_balance_local(S):
    S = np.asarray(S, dtype=int)
    n = S.shape[0]
    c = np.zeros(n, dtype=float)

    for i in range(n):
        balanced = 0
        total = 0

        for j in range(n):
            for k in range(j + 1, n):
                if i != j and i != k:
                    if S[i, j] != 0 and S[j, k] != 0 and S[k, i] != 0:
                        total += 1

                        if S[i, j] * S[j, k] * S[k, i] > 0:
                            balanced += 1

        if total > 0:
            c[i] = balanced / total
        else:
            c[i] = 0.0

    return c

def structural_balance_groups(S):
    S = np.asarray(S, dtype=int)
    n = S.shape[0]

    visited = np.zeros(n, dtype=bool)
    group = -np.ones(n, dtype=int)
    g = 0

    for start in range(n):
        if visited[start]:
            continue

        stack = [start]
        visited[start] = True
        group[start] = g

        while stack:
            u = stack.pop()

            for v in range(n):
                if u != v and S[u, v] == 1 and not visited[v]:
                    visited[v] = True
                    group[v] = g
                    stack.append(v)

        g += 1

    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] == 0:
                continue

            # negative edge inside a group -> not balanced
            if group[i] == group[j] and S[i, j] == -1:
                return False, None, None

            # positive edge between groups -> not balanced
            if group[i] != group[j] and S[i, j] == 1:
                return False, None, None

    clusters = {}
    for i, grp in enumerate(group):
        if grp not in clusters:
            clusters[grp] = []
        clusters[grp].append(i)

    return True, group, clusters

def pearson_matrix(A, mode="out"):
    A = np.asarray(A, dtype=float)

    if mode == "out":
        X = A.T      
    elif mode == "in":
        X = A        
    else:
        raise ValueError("mode must be 'out' or 'in'")

    n = X.shape[0]
    R = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            x = X[i]
            y = X[j]

            x_mean = np.mean(x)
            y_mean = np.mean(y)

            cov = np.sum((x - x_mean) * (y - y_mean))
            sigma_x = np.sqrt(np.sum((x - x_mean)**2))
            sigma_y = np.sqrt(np.sum((y - y_mean)**2))

            if sigma_x == 0 or sigma_y == 0:
                R[i, j] = 0.0
            else:
                R[i, j] = cov / (sigma_x * sigma_y)

    return R

def modularity(A, idx_to_player_id, df, role_col="role"):
    A = np.asarray(A, dtype=float)
    n = len(idx_to_player_id)

    k_in = A.sum(axis=1)
    k_out = A.sum(axis=0)

    m = A.sum()

    if m == 0:
        return np.zeros(n), 0.0

    role_map = (
        df[["player_id", role_col]]
        .dropna()
        .drop_duplicates("player_id")
        .set_index("player_id")[role_col]
        .to_dict()
    )

    roles = [role_map.get(pid, None) for pid in idx_to_player_id]

    c = np.zeros(n, dtype=float)

    for i in range(n):
        if k_in[i] == 0 or roles[i] is None:
            c[i] = 0.0
            continue

        s = 0.0
        for j in range(n):
            if roles[j] is not None and roles[i] == roles[j]:
                s += A[i, j] - (k_in[i] * k_out[j]) / m

        c[i] = s / k_in[i]

    Q_sum = 0.0
    for i in range(n):
        if roles[i] is None:
            continue

        for j in range(n):
            if roles[j] is not None and roles[i] == roles[j]:
                Q_sum += A[i, j] - (k_in[i] * k_out[j]) / (2 * m)

    Q = Q_sum / m

    return c, float(Q)
