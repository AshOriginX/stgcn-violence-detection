import numpy as np
import torch
import torch.nn as nn


# ============================================================
# COCO 17-joint graph used by PySKL
# ============================================================

def normalize_digraph(A):
    Dl = np.sum(A, axis=0)
    Dn = np.zeros_like(A, dtype=np.float32)

    for i in range(len(Dl)):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** -1

    return A @ Dn


def coco_graph():
    num_node = 17

    inward = [
        (15, 13), (13, 11),
        (16, 14), (14, 12),
        (11, 5), (12, 6),
        (9, 7), (7, 5),
        (10, 8), (8, 6),
        (5, 0), (6, 0),
        (1, 0), (3, 1),
        (2, 0), (4, 2),
    ]

    self_link = [(i, i) for i in range(num_node)]
    outward = [(j, i) for i, j in inward]

    def edge2mat(edges):
        A = np.zeros((num_node, num_node), dtype=np.float32)
        for i, j in edges:
            A[j, i] = 1
        return A

    A = np.stack([
        edge2mat(self_link),
        normalize_digraph(edge2mat(inward)),
        normalize_digraph(edge2mat(outward)),
    ])

    return torch.tensor(A, dtype=torch.float32)


# ============================================================
# ST-GCN++ GCN
# ============================================================

class UnitGCN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        A,
        adaptive="init",
        conv_pos="pre",
        with_res=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_subsets = A.size(0)
        self.adaptive = adaptive
        self.conv_pos = conv_pos
        self.with_res = with_res

        if adaptive == "init":
            self.A = nn.Parameter(A.clone())
        else:
            self.register_buffer("A", A)

        if adaptive in ["offset", "importance"]:
            self.PA = nn.Parameter(A.clone())

            if adaptive == "offset":
                nn.init.uniform_(self.PA, -1e-6, 1e-6)
            else:
                nn.init.constant_(self.PA, 1)

        if conv_pos == "pre":
            self.conv = nn.Conv2d(
                in_channels,
                out_channels * self.num_subsets,
                kernel_size=1,
            )
        else:
            self.conv = nn.Conv2d(
                in_channels * self.num_subsets,
                out_channels,
                kernel_size=1,
            )

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if with_res:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.down = nn.Identity()

    def forward(self, x):
        n, c, t, v = x.shape

        # Compute residual from the ORIGINAL input before channel projection.
        if self.with_res:
            res = self.down(x)
        else:
            res = 0

        if self.adaptive == "init":
            A = self.A
        elif self.adaptive == "offset":
            A = self.A + self.PA
        elif self.adaptive == "importance":
            A = self.A * self.PA
        else:
            A = self.A

        if self.conv_pos == "pre":
            x = self.conv(x)
            x = x.view(
                n,
                self.num_subsets,
                -1,
                t,
                v,
            )

            x = torch.einsum(
                "nkctv,kvw->nctw",
                x,
                A,
            )

        else:
            x = torch.einsum(
                "nctv,kvw->nkctw",
                x,
                A,
            )

            x = x.reshape(
                n,
                -1,
                t,
                v,
            )

            x = self.conv(x)

        return self.relu(self.bn(x) + res)


# ============================================================
# Multi-scale Temporal Convolution
# ============================================================

class UnitTCN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=9,
        stride=1,
        dilation=1,
    ):
        super().__init__()

        padding = (
            kernel_size
            + (kernel_size - 1) * (dilation - 1)
            - 1
        ) // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
        )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class MultiScaleTCN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        dropout=0.0,
    ):
        super().__init__()

        ms_cfg = [
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            ("max", 3),
            "1x1",
        ]

        num_branches = len(ms_cfg)

        mid_channels = out_channels // num_branches
        rem_mid_channels = (
            out_channels
            - mid_channels * (num_branches - 1)
        )

        branches = []

        for i, cfg in enumerate(ms_cfg):

            branch_c = (
                rem_mid_channels
                if i == 0
                else mid_channels
            )

            if cfg == "1x1":
                branches.append(
                    nn.Conv2d(
                        in_channels,
                        branch_c,
                        kernel_size=1,
                        stride=(stride, 1),
                    )
                )
                continue

            if cfg[0] == "max":
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels,
                            branch_c,
                            kernel_size=1,
                        ),
                        nn.BatchNorm2d(branch_c),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(
                            kernel_size=(cfg[1], 1),
                            stride=(stride, 1),
                            padding=(1, 0),
                        ),
                    )
                )
                continue

            kernel, dilation = cfg

            branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        branch_c,
                        kernel_size=1,
                    ),
                    nn.BatchNorm2d(branch_c),
                    nn.ReLU(inplace=True),
                    UnitTCN(
                        branch_c,
                        branch_c,
                        kernel_size=kernel,
                        stride=stride,
                        dilation=dilation,
                    ),
                )
            )

        self.branches = nn.ModuleList(branches)

        total_channels = (
            mid_channels * (num_branches - 1)
            + rem_mid_channels
        )

        self.transform = nn.Sequential(
            nn.BatchNorm2d(total_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                total_channels,
                out_channels,
                kernel_size=1,
            ),
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        outputs = [
            branch(x)
            for branch in self.branches
        ]

        x = torch.cat(outputs, dim=1)
        x = self.transform(x)

        return self.drop(self.bn(x))


# ============================================================
# ST-GCN++ block
# ============================================================

class STGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        A,
        stride=1,
        residual=True,
        gcn_adaptive="init",
        gcn_with_res=True,
        tcn_type="mstcn",
        tcn_dropout=0.0,
    ):
        super().__init__()

        self.gcn = UnitGCN(
            in_channels,
            out_channels,
            A,
            adaptive=gcn_adaptive,
            with_res=gcn_with_res,
        )

        if tcn_type != "mstcn":
            raise ValueError(
                "This implementation requires tcn_type='mstcn'"
            )

        self.tcn = MultiScaleTCN(
            out_channels,
            out_channels,
            stride=stride,
            dropout=tcn_dropout,
        )

        self.use_residual = residual

        if not residual:
            self.residual = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = UnitTCN(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.use_residual:
            res = self.residual(x)
        else:
            res = 0

        x = self.gcn(x)
        x = self.tcn(x)

        return self.relu(x + res)


# ============================================================
# ST-GCN++
# ============================================================

class STGCNPlusPlus(nn.Module):

    def __init__(
        self,
        num_classes=2,
        in_channels=2,
        num_person=2,
        base_channels=64,
        ch_ratio=2,
        num_stages=10,
        inflate_stages=(5, 8),
        down_stages=(5, 8),
        tcn_dropout=0.0,
    ):
        super().__init__()

        self.num_person = num_person
        self.num_stages = num_stages

        A = coco_graph()

        self.register_buffer("A", A)

        # Official ST-GCN preprocessing uses VC normalization
        self.data_bn = nn.BatchNorm1d(
            in_channels * 17
        )

        kwargs = dict(
            gcn_adaptive="init",
            gcn_with_res=True,
            tcn_type="mstcn",
            tcn_dropout=tcn_dropout,
        )

        blocks = []

        current_channels = in_channels
        channels = base_channels

        # Initial projection
        blocks.append(
            STGCNBlock(
                current_channels,
                channels,
                A.clone(),
                stride=1,
                residual=False,
                **kwargs,
            )
        )

        current_channels = channels
        inflate_count = 0

        for stage in range(2, num_stages + 1):

            stride = (
                2
                if stage in down_stages
                else 1
            )

            if stage in inflate_stages:
                inflate_count += 1

            out_channels = int(
                base_channels
                * (ch_ratio ** inflate_count)
            )

            blocks.append(
                STGCNBlock(
                    current_channels,
                    out_channels,
                    A.clone(),
                    stride=stride,
                    residual=True,
                    **kwargs,
                )
            )

            current_channels = out_channels

        self.gcn = nn.ModuleList(blocks)

        self.cls_head = nn.Linear(
            current_channels,
            num_classes,
        )

    def forward(self, x):
        """
        Input:
            N,C,T,V,M

        Output:
            N,num_classes
        """

        N, C, T, V, M = x.shape

        # Official ST-GCN format:
        # N,M,T,V,C
        x = x.permute(
            0, 4, 2, 3, 1
        ).contiguous()

        # Normalize across V*C for each person
        x = x.permute(
            0, 1, 3, 4, 2
        ).contiguous()

        x = x.view(
            N * M,
            V * C,
            T,
        )

        x = self.data_bn(x)

        x = x.view(
            N,
            M,
            V,
            C,
            T,
        )

        x = x.permute(
            0, 1, 3, 4, 2
        ).contiguous()

        # Merge persons
        x = x.view(
            N * M,
            C,
            T,
            V,
        )

        for block in self.gcn:
            x = block(x)

        # Restore person dimension
        x = x.view(
            N,
            M,
            x.shape[1],
            x.shape[2],
            x.shape[3],
        )

        # Global temporal + spatial pooling.
        # Preserve the channel dimension.
        x = x.mean(dim=(3, 4))

        # Aggregate persons -> (N, C)
        x = x.mean(dim=1)

        return self.cls_head(x)
