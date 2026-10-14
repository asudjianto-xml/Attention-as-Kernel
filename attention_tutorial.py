"""Run the cross-attention transformer tutorial on the CPU.

Code cells match the notebook. Outputs default to ./outputs.
"""
import matplotlib
matplotlib.use("Agg")


def main():

    from pathlib import Path
    import copy, json, os, platform, time
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import torch
    from torch import nn
    import sklearn
    from sklearn.datasets import fetch_california_housing
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    # Outputs are relative to the working directory unless explicitly configured.
    OUTPUT = Path(os.environ.get('ATTENTION_TUTORIAL_OUTPUT_DIR', 'outputs')).expanduser()
    FIG = OUTPUT / 'figures'
    FIG.mkdir(parents=True, exist_ok=True)
    thread_limits = threadpool_limits(limits=2)
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    plt.rcParams.update({'figure.dpi': 115, 'axes.spines.top': False, 'axes.spines.right': False})
    versions = dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                    matplotlib=matplotlib.__version__, sklearn=sklearn.__version__)
    results = {'versions': versions}
    started = time.time()
    print(versions)

    def savefig(name):
        plt.savefig(FIG / (name + '.pdf'), bbox_inches='tight')
        plt.savefig(FIG / (name + '.png'), bbox_inches='tight', dpi=150)
        plt.show()

    def rmse(y, pred):
        return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred))**2)))

    similarities = np.array([6., 3., 1.])
    values = np.array([10., 20., 40.])
    normalized = similarities / similarities.sum()
    scores = np.log(similarities)
    exp_scores = np.exp(scores - scores.max())
    attention_weights = exp_scores / exp_scores.sum()
    np.testing.assert_allclose(attention_weights, normalized)
    np.testing.assert_allclose(attention_weights @ values, 16.)
    print('Weights:', attention_weights)
    print('Weighted label average:', attention_weights @ values)
    print('Same weights, different values:', attention_weights @ np.array([1., -2., 3.]))

    class CrossAttentionBlock(nn.Module):
        def __init__(self, width=8, heads=2):
            super().__init__()
            assert width % heads == 0
            self.heads, self.head_width = heads, width // heads
            self.norm_attention = nn.LayerNorm(width)
            self.q_map = nn.Linear(width, width)
            self.k_map = nn.Linear(width, width)
            self.v_map = nn.Linear(width, width)
            self.output_map = nn.Linear(width, width)
            self.norm_ffn = nn.LayerNorm(width)
            self.ffn = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(), nn.Linear(2*width, width))

        def split_heads(self, tensor):
            batch, tokens, _ = tensor.shape
            return tensor.reshape(batch, tokens, self.heads, self.head_width).transpose(1, 2)

        def forward(self, tokens, context_size):
            normalized = self.norm_attention(tokens)
            context = normalized[:, :context_size]
            queries = self.split_heads(self.q_map(normalized))
            keys = self.split_heads(self.k_map(context))
            values = self.split_heads(self.v_map(context))
            scores = queries @ keys.transpose(-2, -1) / self.head_width**0.5
            attention = scores.softmax(dim=-1)
            head_output = attention @ values
            combined = head_output.transpose(1, 2).reshape_as(tokens)
            tokens = tokens + self.output_map(combined)
            tokens = tokens + self.ffn(self.norm_ffn(tokens))
            return tokens, attention, values, head_output

    class CrossAttentionTransformer(nn.Module):
        def __init__(self, width=8, heads=2, depth=2):
            super().__init__()
            self.embedding = nn.Linear(3, width)
            self.blocks = nn.ModuleList([CrossAttentionBlock(width, heads) for _ in range(depth)])
            self.readout = nn.Linear(width, 1)

        def forward(self, context_inputs, context_labels, query_inputs):
            mean = context_labels.mean(dim=1, keepdim=True)
            context = torch.stack([context_inputs, context_labels-mean, torch.ones_like(context_inputs)], -1)
            query = torch.stack([query_inputs, torch.zeros_like(query_inputs), torch.zeros_like(query_inputs)], -1)
            tokens = self.embedding(torch.cat([context, query], dim=1))
            context_size = context.shape[1]
            details = []
            for block in self.blocks:
                tokens, attention, values, head_output = block(tokens, context_size)
                details.append((attention, values, head_output))
            prediction = mean + self.readout(tokens[:, context_size:]).squeeze(-1)
            return prediction, details

    # Keep this illustration's random initialization separate from later experiments.
    with torch.random.fork_rng():
        torch.manual_seed(77)
        standard_transformer = CrossAttentionTransformer()
        cx = torch.tensor([[-1., 0., 1.]])
        cy = torch.tensor([[10., 20., 40.]])
        qx = torch.tensor([[-0.5, 0.5]])
        with torch.no_grad():
            prediction, details = standard_transformer(cx, cy, qx)
            assert prediction.shape == (1, 2)
            for attention, values, head_output in details:
                torch.testing.assert_close(attention.sum(-1), torch.ones_like(attention.sum(-1)))
                # First new query, first head: explicitly add its three weighted vectors.
                manual = sum(attention[0, 0, 3, i]*values[0, 0, i] for i in range(3))
                torch.testing.assert_close(manual, head_output[0, 0, 3])
            perm = torch.tensor([2, 0, 1])
            reordered, _ = standard_transformer(cx[:, perm], cy[:, perm], qx)
            alone, _ = standard_transformer(cx, cy, qx[:, :1])
            torch.testing.assert_close(reordered, prediction, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(alone, prediction[:, :1], atol=1e-12, rtol=1e-12)
        # Check that both attention projections and the feed-forward block can learn.
        standard_transformer.zero_grad()
        standard_transformer(cx, cy, qx)[0].square().mean().backward()
        for block in standard_transformer.blocks:
            for parameter in [block.q_map.weight, block.k_map.weight, block.v_map.weight,
                              block.output_map.weight, block.ffn[0].weight]:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                assert parameter.grad.abs().sum() > 0
        print('First layer: weights for new queries (heads x queries x context rows):')
        print(details[0][0][0, :, 3:].numpy().round(3))
        print('Standard block: weighted averages, invariance and gradient checks passed.')

    rng = np.random.default_rng(17)
    def truth(x):
        return np.sin(2.5*x[:, 0]) + 0.35*np.cos(5*x[:, 0])
    def sample(n):
        x = rng.uniform(-2, 2, size=(n, 1))
        return x, truth(x) + rng.normal(0, 0.18, n)
    X, y = sample(48)
    Xval, yval = sample(160)
    Xtest, ytest = sample(240)
    Xgrid = np.linspace(-2.2, 2.2, 400)[:, None]
    ybar = y.mean()
    yc = y - ybar
    plt.figure(figsize=(7, 3))
    plt.scatter(X[:, 0], y, s=20, label='Context labels')
    plt.plot(Xgrid[:, 0], truth(Xgrid), color='black', label='Noise-free curve')
    plt.xlabel('Input'); plt.ylabel('Response'); plt.legend()
    savefig('data')

    def logits(a, b, T):
        return -np.sum((a[:, None, :] - b[None, :, :])**2, axis=-1) / T

    def softmax_rows(z):
        e = np.exp(z - z.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def kernel(a, b, T):
        return np.exp(logits(a, b, T))

    def weights(a, b, T):
        # Subtracting the row maximum prevents exponential underflow.
        return softmax_rows(logits(a, b, T))

    def krr(a, b, labels, T, lam):
        mean = labels.mean()
        alpha = np.linalg.solve(kernel(a, a, T) + lam*np.eye(len(a)), labels-mean)
        return mean + kernel(b, a, T) @ alpha

    T = 0.5  # fixed for the explanatory figures
    K = kernel(X, X, T)
    S = weights(X, X, T)
    Q = weights(Xgrid, X, T)
    assert np.linalg.eigvalsh(K).min() > -1e-10
    np.testing.assert_allclose(S.sum(axis=1), 1)
    np.testing.assert_allclose(ybar + Q @ yc, Q @ y)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    order = np.argsort(X[:, 0])
    im = ax[0].imshow(S[np.ix_(order, order)], aspect='auto', cmap='viridis')
    fig.colorbar(im, ax=ax[0], label='Attention weight')
    ax[0].set(xlabel='Context key (sorted)', ylabel='Context query (sorted)')
    ax[1].scatter(X[:, 0], y, s=12, color='gray')
    ax[1].plot(Xgrid[:, 0], truth(Xgrid), 'k--', label='Truth')
    ax[1].plot(Xgrid[:, 0], krr(X, Xgrid, y, T, .1), label='KRR, penalty 0.1')
    ax[1].plot(Xgrid[:, 0], ybar+Q@yc, label='NW')
    ax[1].set(xlabel='Input', ylabel='Prediction'); ax[1].legend()
    plt.tight_layout(); savefig('krr_nw')

    dot_scores = 2 * Xgrid @ X.T / T - np.sum(X**2, axis=1)[None, :] / T
    Qdot = softmax_rows(dot_scores)
    np.testing.assert_allclose(Qdot, Q, atol=2e-14)
    np.testing.assert_allclose(ybar + Qdot @ yc, ybar + Q @ yc, atol=2e-14)
    print('Gaussian scores and biased dot-product scores agree.')

    def iterate(S, Q, centered_labels, depth, eta=1.0):
        fc = np.zeros_like(centered_labels)
        fq = np.zeros(Q.shape[0])
        context_path, query_path = [], []
        for _ in range(depth):
            residual = centered_labels - fc  # read before either update
            fc = fc + eta * (S @ residual)
            fq = fq + eta * (Q @ residual)
            context_path.append(fc.copy())
            query_path.append(fq.copy())
        return np.array(context_path), np.array(query_path)

    fc_path, fq_path = iterate(S, Q, yc, 1024)
    np.testing.assert_allclose(fq_path[0], Q @ yc, atol=1e-13)
    np.testing.assert_allclose(fq_path[1], Q @ (2*np.eye(len(X))-S) @ yc, atol=1e-13)
    # An independent matrix-power expression checks deeper updates.
    v = sum((np.linalg.matrix_power(np.eye(len(X))-S, j) @ yc for j in range(8)))
    np.testing.assert_allclose(fq_path[7], Q @ v, atol=1e-12)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    for depth in [1, 4, 16, 1024]:
        ax[0].plot(Xgrid[:, 0], ybar+fq_path[depth-1], label=f'{depth} layers')
    ax[0].scatter(X[:, 0], y, s=10, color='gray'); ax[0].legend()
    ax[0].set(xlabel='Input', ylabel='Prediction')
    ax[1].plot(np.arange(1, 1025), np.sqrt(np.mean((yc-fc_path)**2, axis=1)))
    ax[1].set(xlabel='Layers', ylabel='Context RMSE', xscale='log')
    plt.tight_layout(); savefig('depth')

    depths = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    temperatures = [0.02, 0.04, 0.08, 0.2, 0.5, 1.0, 2.0]
    penalties = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
    _, val_path = iterate(S, weights(Xval, X, T), yc, 1024)
    plt.figure(figsize=(6, 3))
    plt.plot(np.arange(1, 1025), [rmse(yval, ybar+p) for p in val_path], label='Validation')
    plt.plot(np.arange(1, 1025), [rmse(y, ybar+p) for p in fc_path], label='Context')
    plt.xlabel('Layers'); plt.ylabel('RMSE'); plt.xscale('log'); plt.legend()
    savefig('validation')

    nw_candidates, it_candidates, kr_candidates = [], [], []
    for temp in temperatures:
        _, path = iterate(weights(X, X, temp), weights(Xval, X, temp), yc, max(depths))
        nw_candidates.append((rmse(yval, ybar+path[0]), temp))
        it_candidates.extend((rmse(yval, ybar+path[d-1]), temp, d) for d in depths)
        kr_candidates.extend((rmse(yval, krr(X, Xval, y, temp, lam)), temp, lam) for lam in penalties)
    _, tn = min(nw_candidates)
    _, ti, di = min(it_candidates)
    _, tk, lk = min(kr_candidates)
    synthetic = {
        'NW': {'T': tn, 'test_rmse': rmse(ytest, ybar+weights(Xtest, X, tn)@yc)},
        'Iterated NW': {'T': ti, 'depth': di, 'test_rmse': rmse(ytest, ybar+iterate(weights(X,X,ti), weights(Xtest,X,ti), yc, di)[1][-1])},
        'KRR': {'T': tk, 'penalty': lk, 'test_rmse': rmse(ytest, krr(X,Xtest,y,tk,lk))}}
    results['synthetic'] = synthetic
    print(json.dumps(synthetic, indent=2))
    results['synthetic_grids'] = {'T': temperatures, 'depth': depths, 'penalty': penalties}
    def grid_edges(parameters, grids):
        return [f"{key}={value:g} ({'lower' if value == min(grids[key]) else 'upper'} edge)"
                for key, value in parameters.items() if key in grids and value in (min(grids[key]), max(grids[key]))]
    for name, row in synthetic.items():
        row['grid_edges'] = grid_edges(row, results['synthetic_grids'])
        print(name, 'grid edges:', row['grid_edges'] or 'none')

    degrees = K.sum(axis=1)
    A = K / np.sqrt(degrees[:, None]*degrees[None, :])
    eig = np.linalg.eigvalsh(A)
    assert eig.min() > -1e-12 and eig.max() <= 1+1e-12
    Sbad = np.array([[0.1, 0.9], [0.9, 0.1]])
    Sasym = np.array([[0.1, 0.9], [0.8, 0.2]])
    np.testing.assert_allclose(softmax_rows(np.log(Sbad)), Sbad)
    r0 = np.array([1., -1.])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    s = np.linspace(0, 1, 300)
    for d in [1, 4, 16, 64]:
        ax[0].plot(s, 1-(1-s)**d, label=f'{d} layers')
    ax[0].set(xlabel='Eigenvalue', ylabel='Fraction fitted'); ax[0].legend()
    for name, mat in [('Symmetric, non-PSD', Sbad), ('Asymmetric', Sasym)]:
        residuals = [np.linalg.norm(np.linalg.matrix_power(np.eye(2)-mat, l)@r0) for l in range(16)]
        ax[1].semilogy(range(16), residuals, label=name)
    ax[1].set(xlabel='Layers', ylabel='Residual norm'); ax[1].legend()
    plt.tight_layout(); savefig('stability')
    print('Non-PSD eigenvalues:', np.linalg.eigvals(Sbad))
    print('Asymmetric eigenvalues:', np.linalg.eigvals(Sasym))

    Xi = np.array([[-1.5], [-0.4], [0.2], [1.4]])
    yi = np.array([1.0, -0.5, 0.8, -0.2])
    mi, ci = yi.mean(), yi-yi.mean()
    Ki = kernel(Xi, Xi, 0.5)
    diag = Ki.sum(axis=1)
    Si = weights(Xi, Xi, 0.5)
    Xqi = np.linspace(-2, 2, 100)[:, None]
    Qi = weights(Xqi, Xi, 0.5)
    limit = mi + Qi @ np.linalg.solve(Si, ci)
    ridgeless = mi + kernel(Xqi, Xi, 0.5) @ np.linalg.solve(Ki, ci)
    np.testing.assert_allclose(mi + Si @ np.linalg.solve(Si, ci), yi)
    np.testing.assert_allclose(mi + Ki @ np.linalg.solve(Ki, ci), yi)
    assert np.max(np.abs(limit-ridgeless)) > 0.01
    _, ip = iterate(Si, Qi, ci, 300)
    np.testing.assert_allclose(mi+ip[-1], limit, atol=1e-10)
    print('Condition number:', np.linalg.cond(Ki))
    print('Largest query difference:', np.max(np.abs(limit-ridgeless)))

    class SmoothingLayer(nn.Module):
        def __init__(self, nw_init=True, T=0.5):
            super().__init__()
            self.q = nn.Linear(3, 2, bias=False)
            self.k = nn.Linear(3, 2, bias=False)
            self.v = nn.Linear(3, 1, bias=False)
            self.T = T
            if nw_init:
                with torch.no_grad():
                    self.q.weight.zero_(); self.k.weight.zero_()
                    self.q.weight[0, 0] = 1; self.k.weight[0, 0] = 1
                    self.v.weight.copy_(torch.tensor([[0., 1., -1.]]))

        def forward(self, context, query):
            old_keys, old_values = self.k(context), self.v(context)
            def update(tokens):
                projected = self.q(tokens)
                scores = -((projected[:, :, None, :] - old_keys[:, None, :, :])**2).sum(-1)/self.T
                correction = scores.softmax(-1) @ old_values
                return torch.cat([tokens[..., :2], tokens[..., 2:3]+correction], dim=-1)
            return update(context), update(query)

    class SmoothingNetwork(nn.Module):
        def __init__(self, depth=4, nw_init=True):
            super().__init__()
            self.layers = nn.ModuleList([SmoothingLayer(nw_init) for _ in range(depth)])
        def forward(self, xc, labels, xq):
            mean = labels.mean(dim=1, keepdim=True)
            centered = labels-mean
            context = torch.stack([xc, centered, torch.zeros_like(xc)], dim=-1)
            query = torch.stack([xq, torch.zeros_like(xq), torch.zeros_like(xq)], dim=-1)
            for layer in self.layers:
                context, query = layer(context, query)
            return mean + query[..., 2]

    net = SmoothingNetwork()
    with torch.no_grad():
        pred = net(torch.tensor(X[:, 0])[None], torch.tensor(y)[None], torch.tensor(Xgrid[:, 0])[None])[0].numpy()
    np.testing.assert_allclose(pred, ybar+fq_path[3], atol=1e-12)
    # Context order and query batch composition must not change the prediction.
    perm = np.random.default_rng(5).permutation(len(X))
    with torch.no_grad():
        reordered = net(torch.tensor(X[perm, 0])[None], torch.tensor(y[perm])[None], torch.tensor(Xgrid[:, 0])[None])[0].numpy()
        alone = net(torch.tensor(X[:, 0])[None], torch.tensor(y)[None], torch.tensor(Xgrid[:1, 0])[None])[0].numpy()
    np.testing.assert_allclose(reordered, pred, atol=1e-12)
    np.testing.assert_allclose(alone, pred[:1], atol=1e-12)
    print('Token network equals four residual NW updates; masking and ordering checks passed.')

    def episodes(seed, batch=32, n=20, m=20):
        gen = np.random.default_rng(seed)
        xc = gen.uniform(-2, 2, (batch, n))
        xq = gen.uniform(-2, 2, (batch, m))
        coeff = gen.normal(size=(batch, 4)) * np.array([1., .5, .3, .5])
        def f(x):
            return coeff[:, 0:1]*np.sin(2*x)+coeff[:, 1:2]*np.cos(3*x)+coeff[:, 2:3]*x+coeff[:, 3:4]
        yc_ep = f(xc)+gen.normal(0, .12, xc.shape)
        yq_ep = f(xq)+gen.normal(0, .12, xq.shape)
        return tuple(torch.tensor(a) for a in (xc, yc_ep, xq, yq_ep))

    validation = episodes(800001, batch=64)
    test = episodes(900001, batch=128)

    def mse_model(model, episode):
        xc, labels, xq, target = episode
        return ((model(xc, labels, xq)-target)**2).mean()

    def train_one(seed, nw_init):
        torch.manual_seed(seed)
        model = SmoothingNetwork(nw_init=nw_init)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        with torch.no_grad():
            best = float(mse_model(model, validation))
        state, best_step = copy.deepcopy(model.state_dict()), 0
        trace = [(0, best)]
        budget_states = {}
        for step in range(1, 1001):
            episode = episodes(seed*10000+step, batch=16)
            optimizer.zero_grad()
            loss = mse_model(model, episode)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % 25 == 0:
                with torch.no_grad():
                    val_loss = float(mse_model(model, validation))
                trace.append((step, val_loss))
                if val_loss < best:
                    best, state, best_step = val_loss, copy.deepcopy(model.state_dict()), step
            if step in (250, 1000):
                budget_states[step] = (copy.deepcopy(state), best_step)
        model.load_state_dict(state)
        model.eval()
        return model, trace, best_step, budget_states

    models, traces, training_rows, budget_rows = {}, {}, [], []
    for seed in [1, 2, 3]:
        for name, init in [('NW initialization', True), ('Random initialization', False)]:
            model, trace, step, budget_states = train_one(seed, init)
            models[(name, seed)], traces[(name, seed)] = model, trace
            with torch.no_grad():
                test_rmse = float(mse_model(model, test).sqrt())
            training_rows.append(dict(seed=seed, initialization=name, selected_step=step, test_rmse=test_rmse))
            # Evaluate only validation-selected checkpoints, not every checkpoint on test data.
            probe = copy.deepcopy(model)
            for budget, (state, selected_step) in budget_states.items():
                probe.load_state_dict(state)
                with torch.no_grad():
                    score = float(mse_model(probe, test).sqrt())
                budget_rows.append(dict(seed=seed, initialization=name, budget=budget,
                                       selected_step=selected_step, test_rmse=score))
    with torch.no_grad():
        baseline_rmse = float(mse_model(SmoothingNetwork(), test).sqrt())
    results['training'] = {'runs': training_rows, 'untrained_iterated_nw_rmse': baseline_rmse,
                           'steps': 1000, 'depth': 4, 'seeds': [1,2,3], 'budget_runs': budget_rows}
    print('Untrained four-layer NW:', baseline_rmse)
    for row in budget_rows:
        print(row)
    plt.figure(figsize=(7, 3))
    for name in ['NW initialization', 'Random initialization']:
        curves = np.array([traces[(name, seed)] for seed in [1,2,3]])
        plt.plot(curves[0,:,0], np.sqrt(curves[:,:,1]).mean(axis=0), label=name)
    plt.xlabel('Training updates'); plt.ylabel('Mean validation RMSE'); plt.legend()
    savefig('training')

    xc, labels, xq, target = test
    shuffle_rng = np.random.default_rng(300001)
    permutations = torch.tensor(np.stack([shuffle_rng.permutation(labels.shape[1]) for _ in range(len(labels))]))
    shuffled_labels = labels.gather(1, permutations)
    diagnostics = []
    with torch.no_grad():
        reference = SmoothingNetwork()(xc, labels, xq)
        for (name, seed), model in models.items():
            original = model(xc, labels, xq)
            shuffled = model(xc, shuffled_labels, xq)
            diagnostics.append(dict(initialization=name, seed=seed,
                test_rmse=float(((original-target)**2).mean().sqrt()),
                shuffled_rmse=float(((shuffled-target)**2).mean().sqrt()),
                prediction_change=float(((shuffled-original)**2).mean().sqrt()),
                departure_from_nw=float(((original-reference)**2).mean().sqrt())))
    results['diagnostics'] = diagnostics
    for name in ['NW initialization', 'Random initialization']:
        rows = [d for d in diagnostics if d['initialization']==name]
        print(name, {k: round(float(np.mean([r[k] for r in rows])), 4)
                     for k in ['test_rmse','shuffled_rmse','prediction_change','departure_from_nw']})

    housing = {'status': 'skipped: California Housing is not cached'}
    try:
        if os.environ.get('ATTENTION_TUTORIAL_SKIP_HOUSING') == '1':
            raise OSError('Housing example disabled')
        data = fetch_california_housing(
            data_home=os.environ.get('ATTENTION_TUTORIAL_DATA_DIR'),
            download_if_missing=False)
    except OSError:
        print(housing['status'])
    else:
        ids = np.random.default_rng(44).permutation(len(data.target))[:768]
        c, v, t = ids[:192], ids[192:384], ids[384:]
        scaler = StandardScaler().fit(data.data[c])
        hx, hv, ht = (scaler.transform(data.data[idx]) for idx in [c,v,t])
        ym, ys = data.target[c].mean(), data.target[c].std()
        hy, hyv, hyt = ((data.target[idx]-ym)/ys for idx in [c,v,t])
        candidates = {'NW': [], 'Iterated NW': [], 'KRR': []}
        housing_grids = {'T': [0.25, 0.5, 1., 4., 16., 64., 256.],
                         'depth': [1,2,4,8,16,32,64,128,256],
                         'penalty': [.001,.01,.1,1.,10.]}
        for temp in housing_grids['T']:
            hs, hq = weights(hx,hx,temp), weights(hv,hx,temp)
            _, hp = iterate(hs,hq,hy-hy.mean(),max(housing_grids['depth']))
            candidates['NW'].append((rmse(hyv,hy.mean()+hp[0]),temp,1))
            candidates['Iterated NW'].extend((rmse(hyv,hy.mean()+hp[d-1]),temp,d) for d in housing_grids['depth'])
            candidates['KRR'].extend((rmse(hyv,krr(hx,hv,hy,temp,lam)),temp,lam) for lam in housing_grids['penalty'])
        rows = {}
        for name, choices in candidates.items():
            score,temp,parameter = min(choices)
            if name == 'KRR':
                prediction = krr(hx,ht,hy,temp,parameter)
            else:
                prediction = hy.mean()+iterate(weights(hx,hx,temp),weights(ht,hx,temp),hy-hy.mean(),int(parameter))[1][-1]
            parameters = {'T': temp}
            if name != 'NW':
                parameters['penalty' if name == 'KRR' else 'depth'] = parameter
            rows[name] = dict(T=temp, parameter=parameter, validation_rmse=score,
                              test_rmse=rmse(hyt,prediction), grid_edges=grid_edges(parameters, housing_grids))
        housing = dict(status='completed', context=192, validation=192, test=384,
                       target_units='context standard deviations', rows=rows, grids=housing_grids)
        print(json.dumps(housing, indent=2))
    results['housing'] = housing

    results['elapsed_seconds'] = time.time()-started
    (OUTPUT / 'results.json').write_text(json.dumps(results, indent=2)+chr(10))
    print(f"Saved results.json and figures in the output directory. Runtime: {results['elapsed_seconds']:.1f} seconds.")


if __name__ == "__main__":
    main()
