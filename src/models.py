import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from layers.hyp_layers import HNNLayer, HyperbolicGraphConvolution, HypLinear  git: https://github.com/HazyResearch/hgcn
from main import device
from dualnet_utils import st_mul_layer, layer_degree_correlation, interlayer_connection_density, \
    interlayer_clustering_coefficient, calculate_average_interlayer_path_length, calculate_interlayer_alignment


class MAC(nn.Module):
    def __init__(self, feats, n_windows, ablation=0):
        super(MAC, self).__init__()
        self.name = 'MAC'
        self.lr = 0.0004
        self.beta = 0.01
        self.n_feats = feats
        self.n_hidden = 32
        self.n_latent = 8
        self.n_window = 5

        self.ablation = ablation
        self.compress_layer = nn.Linear(in_features=feats, out_features=self.feature_out_layer_dime).to(device)
        self.compress_dim = 16

        self.curv_t_hgcn_in = nn.Parameter(torch.Tensor([1.0])).to(device)
        self.curv_t_hgcn_out = nn.Parameter(torch.Tensor([1.0])).to(device)
        self.t_hgcn = HyperbolicGraphConvolution(manifold=self.manifold, in_features=self.feature_out_layer_dime,
                                                 adj_dim=self.n_window, adj_act=nn.Sigmoid(),
                                                 out_features=self.compress_dim, c_in=self.curv_t_hgcn_in,
                                                 c_out=self.curv_t_hgcn_out, dropout=0.0,
                                                 use_bias=False, use_att=0, local_agg=0).to(device)

        self.curv_s_hgcn_in = nn.Parameter(torch.Tensor([1.0])).to(device)
        self.curv_s_hgcn_out = nn.Parameter(torch.Tensor([1.0])).to(device)

        self.s_hgcn = HyperbolicGraphConvolution(manifold=self.manifold, in_features=self.n_window,
                                                 adj_dim=self.feature_out_layer_dime, adj_act=nn.Sigmoid(),
                                                 out_features=self.compress_dim, c_in=self.curv_s_hgcn_in,
                                                 c_out=self.curv_s_hgcn_out, dropout=0.0,
                                                 use_bias=False, use_att=0, local_agg=0).to(device)

        self.curv_out = nn.Parameter(torch.Tensor([1.0])).to(device)
        self.out_layer = nn.Linear(
            in_features=self.compress_dim * self.feature_out_layer_dime + self.compress_dim * self.n_window + self.feature_out_layer_dime ** 2 + self.n_window ** 2,
            out_features=self.compress_dim * self.feature_out_layer_dime + self.compress_dim * self.n_window + self.feature_out_layer_dime ** 2 + self.n_window ** 2).to(
            device)

        self.out_struct_enhance = nn.Sequential(nn.Linear(
            self.n_window * self.n_feats + (self.n_feats * 2) ** 2 + 5,
            self.n_window * self.n_feats),
            nn.ReLU(True)).to(device)
        self.out_struct_enhance_1 = nn.Sequential(nn.Linear(
            self.n_window * self.n_feats + (self.n_feats * 2) ** 2,
            self.n_window * self.n_feats),
            nn.ReLU(True)).to(device)
        # Frequency domain and time domain feature dimension reduction layer
        self.fre_time_fusion_liner = nn.Sequential(nn.Linear(
            self.n_feats * 2,
            self.n_feats),
            nn.ReLU(True)).to(device)
        self.fre_time_fusion_liner_1 = nn.Sequential(nn.Linear(
            self.n_feats,
            self.n_feats),
            nn.ReLU(True)).to(device)
        self.time_liner = nn.Sequential(nn.Linear(
            self.n_window,
            self.n_window),
            nn.ReLU(True)).to(device)

        self.time_gru = nn.GRU(self.n_window, self.n_window).to(device)
        self.fcn_all = nn.Sequential(nn.Linear(
            self.compress_dim * self.feature_out_layer_dime + self.compress_dim * self.n_window + self.feature_out_layer_dime ** 2 + self.n_window ** 2,
            self.n_window * self.n_feats),
            nn.ReLU(True)).to(device)
        self.inter_layer_fusion = nn.Sequential(nn.Linear(
            5, 5), nn.Sigmoid()).to(device)
        self.super_adj_activate = nn.Sequential(nn.Linear(
            self.n_feats * 2, self.n_feats * 2), nn.Sigmoid()).to(device)

        self.t_adj = nn.Parameter(torch.rand(self.n_window, self.n_window, dtype=torch.float64, requires_grad=True)).to(
            device)
        self.s_adj = nn.Parameter(
            torch.zeros(self.feature_out_layer_dime, self.feature_out_layer_dime, dtype=torch.float64,
                        requires_grad=True)).to(device)
        self.t_mask = torch.triu(torch.ones(self.n_window, self.n_window), diagonal=1).to(device) + torch.eye(
            self.n_window).to(device)
        self.s_mask = torch.triu(torch.ones(self.feature_out_layer_dime, self.feature_out_layer_dime), diagonal=1).to(
            device) + torch.eye(self.feature_out_layer_dime).to(device)
        self.t_l = self.n_window ** 2 // 2
        self.s_l = self.n_feats ** 2 // 2
        # manifold, in_features, out_features, c, dropout, use_bias
        self.t_adj_w = HypLinear(manifold=self.manifold, in_features=self.n_window, out_features=self.n_window,
                                 c=self.c, dropout=0.0, use_bias=0).to(device)
        self.s_adj_w = HypLinear(manifold=self.manifold, in_features=self.feature_out_layer_dime,
                                 out_features=self.feature_out_layer_dime, c=self.c, dropout=0.0, use_bias=0).to(device)
        self.ls = None
        lambda_e = 0.1
        for i in range(self.n_window):
            for j in range(self.n_window):
                if i < j:
                    self.t_adj[i][j] = np.exp(-lambda_e * (j - i))
                if i == j:
                    self.t_adj[i][j] = 1.0

        for i in range(self.feature_out_layer_dime):
            for j in range(self.feature_out_layer_dime):
                if i < j:
                    self.s_adj[i][j] = 1.0
                if i == j:
                    self.s_adj[i][j] = 1.0

        # Superadjacency matrix reconstruction
        self.vgae_hiddn_dim = 8
        self.vgae_latent_dim = 8
        self.gc1 = gnn.GCNConv(self.compress_dim, self.vgae_hiddn_dim).to(device)
        self.gc2 = gnn.GCNConv(self.vgae_hiddn_dim, self.vgae_latent_dim).to(device)
        self.dc = gnn.GCNConv(self.vgae_latent_dim, self.compress_dim).to(device)
        self.fc_mu = nn.Linear(self.vgae_latent_dim, self.vgae_latent_dim).to(device)
        self.fc_logvar = nn.Linear(self.vgae_latent_dim, self.vgae_latent_dim).to(device)
        self.inter_top_k = 20
        self.ideal_alignment_matrix = nn.Parameter(torch.ones((self.n_feats * 2, self.n_feats * 2)))

    def vgae_encoder(self, x, edge_index):
        hidden = self.gc1(x, edge_index).relu()
        z = self.gc2(hidden, edge_index)
        mu = self.fc_mu(z)
        logvar = self.fc_logvar(z)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
            self.input_dim, self.output_dim, self.bias, self.c
        )

    def encoder_Feature(self, x, adj):
        x = x.view(-1, self.n_feats)
        h = self.encoder_model.encode(x, adj)
        h = self.decoder_model.decode(h, adj)

        return h

    def generate_adj(self, x):
        x = x.t()
        s_adj = torch.zeros((self.n_feats, self.n_feats), dtype=torch.float64).to(device)
        for i in range(self.n_feats):
            for j in range(i + 1, self.n_feats):
                s_adj[i][j] = self.manifold.sqdist(x[i], x[j], self.c)
        s_adj = s_adj + s_adj.t()
        return x, s_adj

    def update_adj(self):
        with torch.no_grad():
            self.t_adj *= self.t_mask
            self.s_adj *= self.s_mask

    def getSuper_edgeindex(self, adj):
        adj_matrix = torch.tensor(adj)
        row, col = torch.where(adj_matrix > 0)
        edge_index = torch.stack([row, col], dim=0)
        edge_index = torch.sort(edge_index, dim=1)[0]
        unique_edges = edge_index[:, 1:] != edge_index[:, :-1]
        unique_edges = torch.cat([torch.tensor([True]), unique_edges.all(dim=0)])
        edge_index = edge_index[:, unique_edges]
        return edge_index

    def forward(self, x, t_adj_hyp=None, s_adj_hyp=None, fft=None, hidden=None, noise=None, gsa=None):
        if noise is not None:
            x = x + noise
        x = torch.nan_to_num(x, nan=0.)
        if s_adj_hyp is None:
            t_adj_hyp = nn.Parameter(torch.ones(self.n_window, self.n_window), requires_grad=True).to(device).to(
                torch.float64)
            s_adj_hyp = nn.Parameter(torch.ones(self.feature_out_layer_dime, self.feature_out_layer_dime),
                                     requires_grad=True).to(device).to(torch.float64)
        x = x.view(-1, self.n_feats)
        x_fft = fft.view(-1, self.n_feats)
        frequency = torch.concatenate([x, x_fft], dim=1)
        frequency = self.fre_time_fusion_liner(frequency)
        enhance_time = self.time_liner(x.t())
        enhance_time, h = self.time_gru(enhance_time, hidden)
        t_f, t_adj_hyp = self.t_hgcn((frequency, t_adj_hyp))
        s_f, s_adj_hyp = self.s_hgcn((enhance_time, s_adj_hyp))
        x = torch.cat((t_f.view(-1), s_f.view(-1), t_adj_hyp.view(-1), s_adj_hyp.view(-1))).view(1, -1)
        x = self.out_layer(x)
        out = x
        t_adj_hyp = F.sigmoid(out.view(-1)[-(self.n_window ** 2 + self.feature_out_layer_dime ** 2): -(
                self.n_window ** 2 + self.feature_out_layer_dime ** 2) + self.n_window ** 2]).detach()

        s_adj_hyp = F.sigmoid(out.view(-1)[-(self.feature_out_layer_dime ** 2):]).detach()
        t_f = F.sigmoid(out.view(-1)[:self.n_window * self.compress_dim]).detach().view(-1, self.compress_dim)
        s_f = F.sigmoid(out.view(-1)[
                        self.n_window * self.compress_dim: self.n_window * self.compress_dim + self.n_feats * self.compress_dim]).detach().view(
            -1, self.compress_dim)

        adapt_gsl_out = self.fcn_all(out)
        # The dual-layer network structure is restored
        g_1 = t_adj_hyp.view(-1, self.n_window).cpu().detach().numpy()
        g_2 = s_adj_hyp.view(-1, self.n_feats).cpu().detach().numpy()
        g_1_pad = np.zeros(shape=(g_2.shape[0], g_2.shape[0]))
        g_1_pad[:g_1.shape[0], :g_1.shape[0]] = g_1

        g_1_x_pad = np.zeros(shape=(g_2.shape[0], self.compress_dim))
        g_1_x_pad[:g_1.shape[0]] = t_f.cpu().detach().numpy()
        g_2_x = s_f.cpu().detach().numpy()

        super_adj, inter_layer, g_1, g_2 = st_mul_layer(g_1=g_1_pad, g_2=g_2, g_1_x_pad=g_1_x_pad, g_2_x=g_2_x,
                                                        g_1_x=t_f.cpu().detach().numpy(),
                                                        top_k=self.inter_top_k)
        if gsa is not None:
            super_adj = super_adj * gsa.cpu().detach().numpy()

        layer_correlations = layer_degree_correlation([g_1, g_2])
        densities = interlayer_connection_density(super_adj, num_nodes_per_layer=[self.n_feats, self.n_feats])
        clustering_coeffs = interlayer_clustering_coefficient(super_adj,
                                                              num_nodes_per_layer=[self.n_feats, self.n_feats])
        path_lengths = calculate_average_interlayer_path_length(super_adj,
                                                                num_nodes_per_layer=[self.n_feats, self.n_feats])

        alignment_score = calculate_interlayer_alignment(super_adj,
                                                         self.ideal_alignment_matrix.cpu().detach().numpy())
        fusion_inter_layer = torch.FloatTensor(
            np.array([layer_correlations, densities, clustering_coeffs, path_lengths, alignment_score])).to(
            device).to(
            torch.float64)
        fusion_inter_layer = torch.nan_to_num(fusion_inter_layer, nan=0.)
        funsion = self.inter_layer_fusion(fusion_inter_layer)
        funsion = torch.nan_to_num(funsion, nan=0.)

        x = torch.concatenate([torch.FloatTensor(g_1_x_pad).to(device).to(torch.float64),
                               torch.FloatTensor(g_2_x).to(device).to(torch.float64)], dim=0).to(device).to(
            torch.float64)
        edge_index = self.getSuper_edgeindex(super_adj)
        mu, logvar = self.vgae_encoder(x, edge_index.to(device).to(torch.int64))
        z = self.reparametrize(mu, logvar)
        super_adj_generate = F.sigmoid(z @ z.t()).detach()
        x = torch.concatenate([adapt_gsl_out.view(-1), funsion.view(-1), super_adj_generate.view(-1)])
        x = self.out_struct_enhance(x)
        super_adj = torch.tensor(super_adj).to(device).to(torch.float64)
        return x.view(-1), t_adj_hyp.view(self.n_window, -1), s_adj_hyp.view(self.feature_out_layer_dime, -1), \
            torch.stack([self.c,
                         self.curv_liner,
                         self.curv_t_hgcn_in,
                         self.curv_s_hgcn_in,
                         self.curv_t_hgcn_out,
                         self.curv_s_hgcn_out,
                         self.curv_out]), hidden, mu, logvar, super_adj_generate, super_adj, inter_layer, fusion_inter_layer, noise, gsa
