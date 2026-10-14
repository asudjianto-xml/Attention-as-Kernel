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

    # Two projected queries, three projected keys and two value coordinates.
    bridge_queries = np.array([[1., 0.], [0., 1.]])
    bridge_keys = np.array([[np.sqrt(2)*np.log(6), 0.2],
                            [np.sqrt(2)*np.log(3), 0.5], [0., -0.3]])
    bridge_values = np.array([[10., 1.], [20., -2.], [40., 3.]])
    bridge_scores = bridge_queries @ bridge_keys.T / np.sqrt(bridge_keys.shape[1])

    # Form 1: standard dot-product attention, implemented in PyTorch.
    bridge_attention = torch.softmax(torch.tensor(bridge_scores), dim=-1).numpy()
    bridge_attention_output = bridge_attention @ bridge_values

    # Form 2: the same operation as a normalized positive-kernel sum.
    bridge_kernel = np.exp(bridge_scores)
    bridge_kernel_weights = bridge_kernel / bridge_kernel.sum(axis=1, keepdims=True)
    bridge_kernel_output = (bridge_kernel @ bridge_values) / bridge_kernel.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(bridge_attention, bridge_kernel_weights, atol=1e-14)
    np.testing.assert_allclose(bridge_attention_output, bridge_kernel_output, atol=1e-14)
    np.testing.assert_allclose(bridge_attention[0], [0.6,0.3,0.1], atol=1e-14)
    np.testing.assert_allclose(bridge_attention_output[0], [16.,0.3], atol=1e-14)
    print('Attention output:\n', bridge_attention_output)
    print('Kernel smoother output:\n', bridge_kernel_output)
    print('Maximum difference:', np.abs(bridge_attention_output-bridge_kernel_output).max())

    fig, axes = plt.subplots(1,3,figsize=(10,3.1),constrained_layout=True)
    for ax, matrix, title in zip(axes[:2], [bridge_attention,bridge_kernel_weights],
            ['Softmax of dot-product scores', 'Normalized exponential kernel']):
        im=ax.imshow(matrix,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
        for (i,j), value in np.ndenumerate(matrix):
            ax.text(j,i,f'{value:.3f}',ha='center',va='center',fontsize=11)
        ax.set_xticks(range(3),['key 1','key 2','key 3'])
        ax.set_yticks(range(2),['query 1','query 2'])
        ax.set_title(title,fontsize=11)
    positions=np.arange(2)
    axes[2].bar(positions-.16,bridge_attention_output[:,0],width=.32,label='Attention')
    axes[2].bar(positions+.16,bridge_kernel_output[:,0],width=.32,label='Kernel sum')
    axes[2].set_xticks(positions,['query 1','query 2'])
    axes[2].set_title('Same weighted label average',fontsize=11)
    axes[2].set_ylabel('Output, first value coordinate')
    axes[2].legend(fontsize=9)
    savefig('attention_kernel_identity')

    bridge_labels = np.array([10.,20.,40.])
    bridge_current_fit = np.array([12.,18.,35.])
    bridge_residuals = bridge_labels-bridge_current_fit
    bridge_weights = bridge_attention[0].copy()
    bridge_label_output = bridge_weights @ bridge_labels
    bridge_feature_output = bridge_weights @ bridge_values
    bridge_residual_output = bridge_weights @ bridge_residuals
    np.testing.assert_allclose(bridge_label_output,16.)
    np.testing.assert_allclose(bridge_feature_output,[16.,0.3])
    np.testing.assert_allclose(bridge_residual_output,-0.1,atol=1e-14)
    np.testing.assert_allclose(bridge_weights,bridge_attention[0])
    print('Fixed weights:',bridge_weights)
    print('Labels -> prediction:',bridge_label_output)
    print('Feature vectors -> feature average:',bridge_feature_output)
    print('Residuals -> correction:',bridge_residual_output)

    def normalized_scores(scores):
        exponentials=np.exp(scores-scores.max(axis=-1,keepdims=True))
        return exponentials/exponentials.sum(axis=-1,keepdims=True)

    bridge_temperature=2*np.sqrt(bridge_keys.shape[1])
    bridge_distance_scores=-np.sum((bridge_queries[:,None,:]-bridge_keys[None,:,:])**2,axis=-1)/bridge_temperature
    bridge_biased_scores=bridge_scores-np.sum(bridge_keys**2,axis=1)[None,:]/bridge_temperature
    bridge_gaussian_attention=normalized_scores(bridge_distance_scores)
    np.testing.assert_allclose(bridge_gaussian_attention,normalized_scores(bridge_biased_scores),atol=1e-14)
    assert not np.allclose(bridge_gaussian_attention,bridge_attention)
    print('Dot-product weights for query 1:',bridge_attention[0])
    print('Gaussian weights for query 1:',bridge_gaussian_attention[0])
    print('Same values, different output:',bridge_gaussian_attention[0] @ bridge_values)

    bridge_unit_keys=bridge_keys/np.linalg.norm(bridge_keys,axis=1,keepdims=True)
    bridge_unit_dot=bridge_queries @ bridge_unit_keys.T/np.sqrt(bridge_keys.shape[1])
    bridge_unit_distance=-np.sum((bridge_queries[:,None,:]-bridge_unit_keys[None,:,:])**2,axis=-1)/bridge_temperature
    np.testing.assert_allclose(normalized_scores(bridge_unit_dot),normalized_scores(bridge_unit_distance),atol=1e-14)

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
        return softmax_rows(logits(a, b, T))

    T = 0.5
    G = kernel(X, X, T)
    A_context = weights(X, X, T)
    A_query = weights(Xgrid, X, T)
    np.testing.assert_allclose(A_context.sum(axis=1), 1)
    np.testing.assert_allclose(ybar + A_query @ yc, A_query @ y)
    # The Gaussian/dot-product-with-bias identity also holds on these regression inputs.
    dot_scores = 2 * Xgrid @ X.T / T - np.sum(X**2, axis=1)[None, :] / T
    np.testing.assert_allclose(softmax_rows(dot_scores), A_query, atol=2e-14)
    fig, ax = plt.subplots(1,2,figsize=(9,3.4),constrained_layout=True)
    order=np.argsort(X[:,0])
    for query in [-1.2,0.,1.2]:
        row=weights(np.array([[query]]),X,T)[0]
        ax[0].plot(X[order,0],row[order],marker='.',label=f'Query {query:g}')
    ax[0].set(xlabel='Context input',ylabel='Attention weight'); ax[0].legend()
    ax[1].scatter(X[:,0],y,s=14,color='gray',label='Context labels')
    ax[1].plot(Xgrid[:,0],truth(Xgrid),'k--',label='Noise-free curve')
    ax[1].plot(Xgrid[:,0],ybar+A_query@yc,label='Weighted label average')
    ax[1].set(xlabel='Query input',ylabel='Prediction'); ax[1].legend(fontsize=8)
    savefig('nw_attention')

    def iterate(A_context, A_query, centered_labels, depth, eta=1.0):
        fc = np.zeros_like(centered_labels)
        fq = np.zeros(A_query.shape[0])
        context_path, query_path = [], []
        for _ in range(depth):
            residual = centered_labels - fc  # read before either update
            fc = fc + eta * (A_context @ residual)
            fq = fq + eta * (A_query @ residual)
            context_path.append(fc.copy())
            query_path.append(fq.copy())
        return np.array(context_path), np.array(query_path)

    fc_path, fq_path = iterate(A_context, A_query, yc, 1024)
    np.testing.assert_allclose(fq_path[0], A_query @ yc, atol=1e-13)
    np.testing.assert_allclose(fq_path[1], A_query @ (2*np.eye(len(X))-A_context) @ yc, atol=1e-13)
    # An independent matrix-power expression checks deeper updates.
    v = sum((np.linalg.matrix_power(np.eye(len(X))-A_context, j) @ yc for j in range(8)))
    np.testing.assert_allclose(fq_path[7], A_query @ v, atol=1e-12)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    for depth in [1, 4, 16, 1024]:
        ax[0].plot(Xgrid[:, 0], ybar+fq_path[depth-1], label=f'{depth} layers')
    ax[0].scatter(X[:, 0], y, s=10, color='gray'); ax[0].legend()
    ax[0].set(xlabel='Input', ylabel='Prediction')
    ax[1].plot(np.arange(1, 1025), np.sqrt(np.mean((yc-fc_path)**2, axis=1)))
    ax[1].set(xlabel='Layers', ylabel='Context RMSE', xscale='log')
    plt.tight_layout(); savefig('depth')

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

    # Inspect the actual standard transformer from the preceding cell.
    # These are untrained representations, not fitted predictions.
    from matplotlib.patches import FancyBboxPatch, Circle
    from matplotlib.colors import ListedColormap

    with torch.no_grad():
        tf_mean = cy.mean(dim=1, keepdim=True)
        tf_context = torch.stack([cx, cy-tf_mean, torch.ones_like(cx)], -1)
        tf_queries = torch.stack([qx, torch.zeros_like(qx), torch.zeros_like(qx)], -1)
        tf_tokens = standard_transformer.embedding(torch.cat([tf_context, tf_queries], dim=1))
        tf_states = [tf_tokens[0].numpy().copy()]
        tf_attentions = []
        for tf_block in standard_transformer.blocks:
            tf_tokens, tf_weights, _, _ = tf_block(tf_tokens, cx.shape[1])
            tf_states.append(tf_tokens[0].numpy().copy())
            tf_attentions.append(tf_weights[0].numpy().copy())
        tf_prediction = tf_mean + standard_transformer.readout(tf_tokens[:, cx.shape[1]:]).squeeze(-1)
        torch.testing.assert_close(tf_prediction, standard_transformer(cx, cy, qx)[0])
        tf_first = standard_transformer.blocks[0]
        tf_normalized = tf_first.norm_attention(torch.tensor(tf_states[0])[None])
        tf_q = tf_first.split_heads(tf_first.q_map(tf_normalized))[0, 0, 3:4].numpy()
        tf_k = tf_first.split_heads(tf_first.k_map(tf_normalized[:, :3]))[0, 0].numpy()
        tf_v = tf_first.split_heads(tf_first.v_map(tf_normalized[:, :3]))[0, 0].numpy()
        tf_scores = tf_q @ tf_k.T / tf_first.head_width**0.5
        tf_a = np.exp(tf_scores-tf_scores.max())
        tf_a /= tf_a.sum()
        tf_o = tf_a @ tf_v
    np.testing.assert_allclose(tf_a[0], tf_attentions[0][0, 3], atol=1e-12)
    np.testing.assert_allclose(tf_o, sum(tf_a[0,i]*tf_v[i:i+1] for i in range(3)), atol=1e-12)

    def annotated_matrix(ax, matrix, title, row_labels, col_labels, probability=False, limit=None):
        bound = limit if limit is not None else max(float(np.abs(matrix).max()), 1e-8)
        im = ax.imshow(matrix, cmap='YlGnBu' if probability else 'RdBu_r',
                       vmin=0 if probability else -bound, vmax=1 if probability else bound,
                       aspect='auto')
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_xticks(range(len(col_labels)), col_labels, fontsize=9)
        ax.set_yticks(range(len(row_labels)), row_labels, fontsize=9)
        for (i,j), value in np.ndenumerate(matrix):
            color = 'white' if (value > .65 if probability else abs(value) > .65*bound) else '#182b3a'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=9, color=color)
        return im

    # 1. Query/key projections -> scores -> weights -> value average.
    fig, axes = plt.subplots(2, 3, figsize=(9, 4.8), constrained_layout=True)
    features = ['1','2','3','4']
    context_names = ['c1','c2','c3']
    annotated_matrix(axes[0,0], tf_q, r'1. Query projection: $q$', ['q1'], features)
    annotated_matrix(axes[0,1], tf_k, r'2. Key projections: $K$', context_names, features)
    annotated_matrix(axes[0,2], tf_scores, r'3. Scores: $qK^T/\sqrt{d_k}$', ['q1'], context_names)
    annotated_matrix(axes[1,0], tf_a, '4. Weights: softmax(scores)', ['q1'], context_names, probability=True)
    annotated_matrix(axes[1,1], tf_v, r'5. Value projections: $V$', context_names, features)
    annotated_matrix(axes[1,2], tf_o, r'6. Head output: $aV$', ['q1'], features)
    fig.suptitle('One query, one head: inspect every part of the weighted average', fontsize=13)
    savefig('transformer_qkv')

    # 2. Allowed information flow and the actual weights implementing it.
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    allowed = np.zeros((5,5), dtype=int)
    allowed[:3,:3] = 1
    allowed[3:,:3] = 2
    axes[0].imshow(allowed, cmap=ListedColormap(['#e2e6ea','#b9d7eb','#fed69a']), vmin=0, vmax=2)
    for i in range(5):
        for j in range(5):
            axes[0].text(j,i,'self' if i<3 and j<3 else ('cross' if j<3 else 'blocked'),
                         ha='center', va='center', fontsize=8)
    axes[0].set_xticks(range(5), ['c1','c2','c3','q1','q2'])
    axes[0].set_yticks(range(5), ['c1','c2','c3','q1','q2'])
    axes[0].set_title('Who can supply information?')
    axes[0].set_xlabel('Potential keys / values')
    axes[0].set_ylabel('Tokens making queries')
    axes[0].axhline(2.5,color='white',lw=2); axes[0].axvline(2.5,color='white',lw=2)
    im=annotated_matrix(axes[1],tf_attentions[0][0], 'Actual weights: layer 1, head 1',
                        ['c1','c2','c3','q1','q2'],context_names,probability=True)
    axes[1].axhline(2.5,color='black',lw=1.5)
    axes[1].set_xlabel('Context keys / values only')
    fig.colorbar(im,ax=axes[1],label='Weight',shrink=.75)
    savefig('transformer_context_query')

    # 3. The exact pre-normalization block used in the notebook.
    fig, ax = plt.subplots(figsize=(9, 4.7))
    ax.set(xlim=(0,10),ylim=(0,5)); ax.axis('off')
    def tf_box(x,y,text,width=1.45,height=.72,color='#e4eef7'):
        ax.add_patch(FancyBboxPatch((x-width/2,y-height/2),width,height,
                                   boxstyle='round,pad=0.05',facecolor=color,edgecolor='#29455d',lw=1.2))
        ax.text(x,y,text,ha='center',va='center',fontsize=9)
    def tf_arrow(start,end,color='#29455d'):
        ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',lw=1.5,color=color))
    def tf_add(x,y):
        ax.add_patch(Circle((x,y),.22,facecolor='#fff0d9',edgecolor='#29455d',lw=1.2))
        ax.text(x,y,'+',ha='center',va='center',fontsize=16)
    tf_box(.9,3.45,'Old tokens\ncontext + query',width=1.5)
    tf_box(2.65,3.45,'Layer norm',width=1.35)
    tf_box(4.45,3.45,'Multi-head\nattention',width=1.65)
    tf_box(6.5,3.45,'Concatenate\nthen project',width=1.6)
    tf_add(8.5,3.45)
    tf_arrow((1.7,3.45),(1.92,3.45)); tf_arrow((3.38,3.45),(3.55,3.45))
    tf_arrow((5.33,3.45),(5.62,3.45)); tf_arrow((7.35,3.45),(8.24,3.45))
    # First skip: the original tokens bypass all attention operations.
    ax.plot([.9,.9,8.5],[3.88,4.55,4.55],color='#bf6a24',lw=1.5)
    tf_arrow((8.5,4.55),(8.5,3.69),color='#bf6a24')
    ax.text(4.8,4.68,'Attention skip: retain the old token state',ha='center',fontsize=10,color='#914816')
    ax.text(4.45,2.65,'Queries: all tokens\nKeys and values: old context only',ha='center',fontsize=9)
    tf_box(8.5,1.2,'Layer norm',width=1.35)
    tf_box(6.5,1.2,'Feed-forward\nnetwork',width=1.6)
    tf_add(3.9,1.2)
    tf_box(1.55,1.2,'Updated tokens',width=1.7,color='#e3f1e5')
    tf_arrow((8.5,3.20),(8.5,1.63)); tf_arrow((7.75,1.2),(7.35,1.2))
    tf_arrow((5.65,1.2),(4.16,1.2)); tf_arrow((3.65,1.2),(2.46,1.2))
    # Second skip carries the result of the attention residual addition.
    ax.plot([8.5,3.9],[2.15,2.15],color='#bf6a24',lw=1.5)
    tf_arrow((3.9,2.15),(3.9,1.44),color='#bf6a24')
    ax.text(6.0,2.28,'Feed-forward skip',ha='center',fontsize=10,color='#914816')
    ax.text(6.5,.48,'Transforms each token separately',ha='center',fontsize=9)
    ax.text(1.55,.48,'Repeat for the next layer',ha='center',fontsize=9)
    savefig('transformer_block')

    # 4. Two heads can form different averages, and those weights can change by layer.
    fig, axes = plt.subplots(2,2,figsize=(8,5.2),constrained_layout=True)
    for layer in range(2):
        for head in range(2):
            im=annotated_matrix(axes[layer,head],tf_attentions[layer][head,3:],
                                f'Layer {layer+1}, head {head+1}', ['q1','q2'],context_names,probability=True)
            axes[layer,head].set_xlabel('Context keys')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Attention weight',shrink=.8)
    fig.suptitle('Query-to-context attention in the untrained standard transformer',fontsize=12)
    savefig('transformer_heads')

    # 5. Token vectors after embedding and after each complete block.
    fig, axes = plt.subplots(2,3,figsize=(10,5.2),constrained_layout=True)
    for group, rows in enumerate([slice(0,3),slice(3,5)]):
        state_limit=max(float(np.abs(state[rows]).max()) for state in tf_states)
        for layer in range(3):
            ax=axes[group,layer]
            im=ax.imshow(tf_states[layer][rows],cmap='RdBu_r',vmin=-state_limit,vmax=state_limit,aspect='auto')
            if group==0:
                ax.set_title('After embedding' if layer==0 else f'After block {layer}')
            ax.set_xticks(range(8),range(1,9))
            names=['c1','c2','c3'] if group==0 else ['q1','q2']
            ax.set_yticks(range(len(names)),names)
            ax.set_xlabel('Token coordinate')
            if layer==0:
                ax.set_ylabel('Context tokens' if group==0 else 'Query tokens')
        fig.colorbar(im,ax=axes[group,:].tolist(),label='Context scale' if group==0 else 'Query scale',shrink=.85)
    fig.suptitle('Token states across blocks: a shared scale within each row of panels',fontsize=12)
    savefig('transformer_states')

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

    def krr(a, b, labels, T, lam):
        mean = labels.mean()
        alpha = np.linalg.solve(kernel(a, a, T) + lam*np.eye(len(a)), labels-mean)
        return mean + kernel(b, a, T) @ alpha

    assert np.linalg.eigvalsh(G).min() > -1e-10
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    order = np.argsort(X[:, 0])
    im = ax[0].imshow(A_context[np.ix_(order, order)], aspect='auto', cmap='viridis')
    fig.colorbar(im, ax=ax[0], label='Attention weight')
    ax[0].set(xlabel='Context key (sorted)', ylabel='Context query (sorted)')
    ax[1].scatter(X[:, 0], y, s=12, color='gray')
    ax[1].plot(Xgrid[:, 0], truth(Xgrid), 'k--', label='Truth')
    ax[1].plot(Xgrid[:, 0], krr(X, Xgrid, y, T, .1), label='KRR, penalty 0.1')
    ax[1].plot(Xgrid[:, 0], ybar+A_query@yc, label='NW')
    ax[1].set(xlabel='Input', ylabel='Prediction'); ax[1].legend()
    plt.tight_layout(); savefig('krr_nw')

    depths = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    temperatures = [0.02, 0.04, 0.08, 0.2, 0.5, 1.0, 2.0]
    penalties = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
    _, val_path = iterate(A_context, weights(Xval, X, T), yc, 1024)
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

    degrees = G.sum(axis=1)
    B = G / np.sqrt(degrees[:, None]*degrees[None, :])
    eig = np.linalg.eigvalsh(B)
    assert eig.min() > -1e-12 and eig.max() <= 1+1e-12
    A_bad = np.array([[0.1, 0.9], [0.9, 0.1]])
    A_asym = np.array([[0.1, 0.9], [0.8, 0.2]])
    np.testing.assert_allclose(softmax_rows(np.log(A_bad)), A_bad)
    r0 = np.array([1., -1.])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    s = np.linspace(0, 1, 300)
    for d in [1, 4, 16, 64]:
        ax[0].plot(s, 1-(1-s)**d, label=f'{d} layers')
    ax[0].set(xlabel='Eigenvalue', ylabel='Fraction fitted'); ax[0].legend()
    for name, mat in [('Symmetric, non-PSD', A_bad), ('Asymmetric', A_asym)]:
        residuals = [np.linalg.norm(np.linalg.matrix_power(np.eye(2)-mat, l)@r0) for l in range(16)]
        ax[1].semilogy(range(16), residuals, label=name)
    ax[1].set(xlabel='Layers', ylabel='Residual norm'); ax[1].legend()
    plt.tight_layout(); savefig('stability')
    print('Non-PSD eigenvalues:', np.linalg.eigvals(A_bad))
    print('Asymmetric eigenvalues:', np.linalg.eigvals(A_asym))

    Xi = np.array([[-1.5], [-0.4], [0.2], [1.4]])
    yi = np.array([1.0, -0.5, 0.8, -0.2])
    mi, ci = yi.mean(), yi-yi.mean()
    G_small = kernel(Xi, Xi, 0.5)
    diag = G_small.sum(axis=1)
    A_context_small = weights(Xi, Xi, 0.5)
    Xqi = np.linspace(-2, 2, 100)[:, None]
    A_query_small = weights(Xqi, Xi, 0.5)
    limit = mi + A_query_small @ np.linalg.solve(A_context_small, ci)
    ridgeless = mi + kernel(Xqi, Xi, 0.5) @ np.linalg.solve(G_small, ci)
    np.testing.assert_allclose(mi + A_context_small @ np.linalg.solve(A_context_small, ci), yi)
    np.testing.assert_allclose(mi + G_small @ np.linalg.solve(G_small, ci), yi)
    assert np.max(np.abs(limit-ridgeless)) > 0.01
    _, ip = iterate(A_context_small, A_query_small, ci, 300)
    np.testing.assert_allclose(mi+ip[-1], limit, atol=1e-10)
    print('Condition number:', np.linalg.cond(G_small))
    print('Largest query difference:', np.max(np.abs(limit-ridgeless)))

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
