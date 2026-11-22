from typing import Any
import logging
import time
import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch.nn import Sequential, Dropout, Linear
import torch.nn.functional as F
from torch import nn
from plyfile import PlyData, PlyElement
import numpy as np
import copy

from torch_points3d.core.common_modules import FastBatchNorm1d
from torch_points3d.modules.KPConv import *
from torch_points3d.core.base_conv.partial_dense import *
from torch_points3d.core.common_modules import MultiHeadClassifier
from torch_points3d.models.base_model import BaseModel
from torch_points3d.models.base_architectures.unet import UnwrappedUnetBasedModel
from torch_points3d.datasets.multiscale_data import MultiScaleBatch
from torch_geometric.data import Data
from torch_geometric.nn import knn

from torch_points3d.datasets.change_detection.pair import PairBatch, PairMultiScaleBatch
from torch_points3d.models.change_detection.absh import AuxiliaryBalancedLoss
from torch_points3d.models.change_detection.lmsp import LMSP
from torch_points3d.models.change_detection.ugsp import UGSP


log = logging.getLogger(__name__)

class BaseFactoryPSI:
    def __init__(self, module_name_down_1, module_name_down_2, module_name_up_1, module_name_up_2, modules_lib):
        self.module_name_down_1 = module_name_down_1
        self.module_name_down_2 = module_name_down_2
        self.module_name_up_1 = module_name_up_1
        self.module_name_up_2 = module_name_up_2
        self.modules_lib = modules_lib

    def get_module(self, flow):
        if flow.upper() == "UP":
            if "1" in flow:
                return getattr(self.modules_lib, self.module_name_up_1, None)
            else:
                return getattr(self.modules_lib, self.module_name_up_2, None)
        elif "1" in flow:
            return getattr(self.modules_lib, self.module_name_down_1, None)
        else:
            return getattr(self.modules_lib, self.module_name_down_2, None)


####################SIAMESE ENCODER FUSION KP CONV ############################
class SiamEncFusionKPConv(UnwrappedUnetBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        # Extract parameters from the dataset
        self.threshold_dis_init = 0.5
        self._num_classes = dataset.num_classes
        self._weight_classes = dataset.weight_classes
        try:
            self._ignore_label = dataset.ignore_label
        except:
            self._ignore_label = None
        self._use_category = getattr(option, "use_category", False)
        if self._use_category:
            if not dataset.class_to_segments:
                raise ValueError(
                    "The dataset needs to specify a class_to_segments property when using category information for segmentation"
                )
            self._class_to_seg = dataset.class_to_segments
            self._num_categories = len(self._class_to_seg)
            log.info("Using category information for the predictions with %i categories", self._num_categories)
        else:
            self._num_categories = 0

        # Assemble encoder / decoder
        opt = copy.deepcopy(option)
        super(UnwrappedUnetBasedModel, self).__init__(opt)
        self._spatial_ops_dict = {"neighbour_finder": [], "sampler": [], "upsample_op": []}
        self._init_from_compact_format(opt, model_type, dataset, modules)

        # Unshared weight :  2 down modules
        # Build final MLP
        last_mlp_opt = option.mlp_cls
        if self._use_category:
            self.FC_layer = MultiHeadClassifier(
                last_mlp_opt.nn[0],
                self._class_to_seg,
                dropout_proba=last_mlp_opt.dropout,
                bn_momentum=last_mlp_opt.bn_momentum,
            )
        else:
            in_feat = last_mlp_opt.nn[0] + self._num_categories
            self.FC_layer = Sequential()
            for i in range(1, len(last_mlp_opt.nn)):
                self.FC_layer.add_module(
                    str(i),
                    Sequential(
                        *[
                            Linear(in_feat, last_mlp_opt.nn[i], bias=False),
                            FastBatchNorm1d(last_mlp_opt.nn[i], momentum=last_mlp_opt.bn_momentum),
                            nn.LeakyReLU(0.2),
                        ]
                    ),
                )
                in_feat = last_mlp_opt.nn[i]

            if last_mlp_opt.dropout:
                self.FC_layer.add_module("Dropout", Dropout(p=last_mlp_opt.dropout))

            self.FC_layer.add_module("Class", Lin(in_feat, self._num_classes, bias=False))
            self.FC_layer.add_module("Softmax", nn.LogSoftmax(-1))
        self.lambda_reg = self.get_from_opt(option, ["loss_weights", "lambda_reg"])
        self.lambda_internal_losses = self.get_from_opt(option, ["loss_weights", "lambda_internal_losses"])
        self.visual_names = ["data_visual"]

        class_counts = self._extract_class_counts(dataset, self._num_classes)

        # LMSP models the two parent distributions with class-conditional GMMs.
        lmsp_opt = getattr(option, "lmsp", None)
        self.lmsp_enabled = bool(getattr(lmsp_opt, "enabled", True)) if lmsp_opt is not None else True
        self.lmsp_fusion_alpha = (
            float(getattr(lmsp_opt, "fusion_alpha", 0.5)) if lmsp_opt is not None else 0.5
        )
        if not 0.0 <= self.lmsp_fusion_alpha <= 1.0:
            raise ValueError("LMSP fusion_alpha must be in [0, 1]")
        if self.lmsp_enabled:
            deep_feature_dim = int(opt.down_conv_1.down_conv_nn[-1][-1][-1])
            unchanged_labels = getattr(lmsp_opt, "unchanged_labels", [0]) if lmsp_opt is not None else [0]
            self.lmsp = LMSP(
                feature_dim=deep_feature_dim,
                num_components=int(getattr(lmsp_opt, "num_components", 3)) if lmsp_opt is not None else 3,
                em_iterations=int(getattr(lmsp_opt, "em_iterations", 5)) if lmsp_opt is not None else 5,
                momentum=float(getattr(lmsp_opt, "momentum", 0.1)) if lmsp_opt is not None else 0.1,
                min_samples=int(getattr(lmsp_opt, "min_samples", 8)) if lmsp_opt is not None else 8,
                variance_floor=float(getattr(lmsp_opt, "variance_floor", 1e-3))
                if lmsp_opt is not None
                else 1e-3,
                unchanged_labels=unchanged_labels,
                ignore_label=self._ignore_label,
                class_counts=class_counts,
            )
        else:
            self.lmsp = None

        # ABSH keeps the original full-data head and adds an independent head.
        absh_opt = getattr(option, "absh", None)
        self.absh_enabled = bool(getattr(absh_opt, "enabled", True)) if absh_opt is not None else True
        self.lambda_bal = float(getattr(absh_opt, "lambda_bal", 1.0)) if absh_opt is not None else 1.0
        configured_counts = getattr(absh_opt, "class_counts", None) if absh_opt is not None else None
        if configured_counts is not None:
            class_counts = torch.as_tensor(list(configured_counts), dtype=torch.float32)
        if class_counts.numel() != self._num_classes:
            raise ValueError("ABSH class_counts must contain one count per output class")
        if self.absh_enabled:
            self.FC_layer_aux = copy.deepcopy(self.FC_layer)
            self.absh_loss = AuxiliaryBalancedLoss(class_counts, ignore_label=self._ignore_label)
        else:
            self.FC_layer_aux = None
            self.absh_loss = None

        self.loss_names = ["loss_cd", "loss_main"]
        if self.absh_enabled:
            self.loss_names.append("loss_bal")
        if self.lambda_reg:
            self.loss_names.append("loss_reg")

        # UGSP is a one-way P0 -> P1 prior.  The original change mask is still
        # used to build the input features so enabling UGSP changes only CADA.
        ugsp_opt = getattr(option, "ugsp", None)
        self.ugsp_enabled = bool(getattr(ugsp_opt, "enabled", True)) if ugsp_opt is not None else True
        self.ugsp_stats_every = int(getattr(ugsp_opt, "stats_every", 0)) if ugsp_opt is not None else 0
        self._ugsp_forward_count = 0
        gate_floor = float(getattr(ugsp_opt, "gate_floor", 0.1)) if ugsp_opt is not None else 0.1
        if self.ugsp_enabled:
            self.ugsp = UGSP(
                radius=float(getattr(ugsp_opt, "radius", 2.5)) if ugsp_opt is not None else 2.5,
                max_neighbors=int(getattr(ugsp_opt, "max_neighbors", 48)) if ugsp_opt is not None else 48,
                max_candidates=int(getattr(ugsp_opt, "max_candidates", 64)) if ugsp_opt is not None else 64,
                alpha=float(getattr(ugsp_opt, "alpha", 0.5)) if ugsp_opt is not None else 0.5,
                graph_k=int(getattr(ugsp_opt, "graph_k", 8)) if ugsp_opt is not None else 8,
                rho=getattr(ugsp_opt, "rho", None) if ugsp_opt is not None else None,
                h=getattr(ugsp_opt, "h", None) if ugsp_opt is not None else None,
                eta=float(getattr(ugsp_opt, "eta", 1.0)) if ugsp_opt is not None else 1.0,
                xi=float(getattr(ugsp_opt, "xi", 0.5)) if ugsp_opt is not None else 0.5,
                omega=float(getattr(ugsp_opt, "omega", 0.5)) if ugsp_opt is not None else 0.5,
                registration_error=float(getattr(ugsp_opt, "registration_error", 0.1))
                if ugsp_opt is not None
                else 0.1,
                beta=float(getattr(ugsp_opt, "beta", 4.0)) if ugsp_opt is not None else 4.0,
            )
        else:
            self.ugsp = None

        # Build Difference Module
        self.batch_size = 10
        self.self_att_list = nn.ModuleList()
        # self_att = Attention(in_feat)
        self.diff_conv_list = Sequential()
        self.pos_xyz_list = Sequential()
        init_dim = 128
        up_dim_init = 512
        self.k_n = [1, 1, 1, 1, 1]
        for i in range(len(opt.down_conv_1.down_conv_nn)):
            # self_att = Attention(int(up_dim_init / pow(2, i)))
            self_att = Attention(int(init_dim * pow(2, i)), gate_floor=gate_floor)
            # self.self_att_list.append([self_att1, self_att2])
            self.self_att_list.append(self_att)
            self.diff_conv_list.add_module(
                str(i),
                Sequential(
                    *[
                        Linear(init_dim * pow(2, i) * 2, init_dim * pow(2, i) // 2, bias=False),
                        FastBatchNorm1d(init_dim * pow(2, i) // 2),
                        nn.LeakyReLU(0.2),
                    ]
                ),
            )
            self.pos_xyz_list.add_module(
                str(i),
                Sequential(
                    *[
                        Linear(6, init_dim * pow(2, i) // 2, bias=False),
                        FastBatchNorm1d(init_dim * pow(2, i) // 2),
                        nn.LeakyReLU(0.2),
                    ]
                ),
            )


    def _init_from_compact_format(self, opt, model_type, dataset, modules_lib):
        """Create a unetbasedmodel from the compact options format - where the
        same convolution is given for each layer, and arguments are given
        in lists
        """
        self.down_modules_1 = nn.ModuleList()
        self.down_modules_2 = nn.ModuleList()
        self.inner_modules = nn.ModuleList()
        self.up_modules_1 = nn.ModuleList()
        self.up_modules_2 = nn.ModuleList()

        self.save_sampling_id_1 = opt.down_conv_1.get('save_sampling_id')
        self.save_sampling_id_2 = opt.down_conv_2.get('save_sampling_id')

        # Factory for creating up and down modules
        factory_module_cls = self._get_factory(model_type, modules_lib)
        down_conv_cls_name_1 = opt.down_conv_1.module_name
        down_conv_cls_name_2 = opt.down_conv_2.module_name
        up_conv_cls_name_1 = opt.up_conv_1.module_name if opt.get('up_conv_1') is not None else None
        up_conv_cls_name_2 = opt.up_conv_2.module_name if opt.get('up_conv_2') is not None else None
        self._factory_module = factory_module_cls(
            down_conv_cls_name_1, down_conv_cls_name_2, up_conv_cls_name_1, up_conv_cls_name_2, modules_lib
        )  # Create the factory object

        # Loal module
        contains_global = hasattr(opt, "innermost") and opt.innermost is not None
        if contains_global:
            inners = self._create_inner_modules(opt.innermost, modules_lib)
            for inner in inners:
                self.inner_modules.append(inner)
        else:
            self.inner_modules.append(Identity())

        # Down modules
        for i in range(len(opt.down_conv_1.down_conv_nn)):
            args = self._fetch_arguments(opt.down_conv_1, i, "DOWN_1")
            conv_cls = self._get_from_kwargs(args, "conv_cls")
            down_module = conv_cls(**args)
            self._save_sampling_and_search(down_module)
            self.down_modules_1.append(down_module)
        for i in range(len(opt.down_conv_2.down_conv_nn)):
            args = self._fetch_arguments(opt.down_conv_2, i, "DOWN_2")
            conv_cls = self._get_from_kwargs(args, "conv_cls")
            down_module = conv_cls(**args)
            self._save_sampling_and_search(down_module)
            self.down_modules_2.append(down_module)

        # Up modules
        if up_conv_cls_name_1:
            for i in range(len(opt.up_conv_1.up_conv_nn)):
                args = self._fetch_arguments(opt.up_conv_1, i, "UP")
                conv_cls = self._get_from_kwargs(args, "conv_cls")
                up_module = conv_cls(**args)
                self._save_upsample(up_module)
                self.up_modules_1.append(up_module)
        if up_conv_cls_name_2:
            for i in range(len(opt.up_conv_2.up_conv_nn)):
                args = self._fetch_arguments(opt.up_conv_2, i, "UP")
                conv_cls = self._get_from_kwargs(args, "conv_cls")
                up_module = conv_cls(**args)
                self._save_upsample(up_module)
                self.up_modules_2.append(up_module)

        self.metric_loss_module, self.miner_module = BaseModel.get_metric_loss_and_miner(
            getattr(opt, "metric_loss", None), getattr(opt, "miner", None)
        )

    def _get_factory(self, model_name, modules_lib) -> BaseFactoryPSI:
        factory_module_cls = getattr(modules_lib, "{}Factory".format(model_name), None)
        if factory_module_cls is None:
            factory_module_cls = BaseFactoryPSI
        return factory_module_cls

    def set_input(self, data, device):
        data = data.to(device)
        data.x = add_ones(data.pos, data.x, True)
        self.batch_idx = data.batch
        if isinstance(data, PairMultiScaleBatch):
            self.pre_computed = data.multiscale
            self.upsample = data.upsample
        else:
            self.pre_computed = None
            self.upsample = None
        if getattr(data, "pos_target", None) is not None:
            data.x_target = add_ones(data.pos_target, data.x_target, True)
            if isinstance(data, PairMultiScaleBatch):
                self.pre_computed_target = data.multiscale_target
                self.upsample_target = data.upsample_target
                del data.multiscale_target
                del data.upsample_target
            else:
                self.pre_computed_target = None
                self.upsample_target = None

            self.input0, self.input1 = data.to_data()
            self.batch_idx_target = data.batch_target
            self.labels0 = data.y.to(device)
            self.labels1 = data.y_target.to(device)
            self.labels = [self.labels0, self.labels1]
        else:
            self.input = data
            self.labels0 = None
            self.labels1 = None
            self.labels = [self.labels0, self.labels1]

    def forward(self, *args, **kwargs) -> Any:
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        stack_down = []
        encoder_levels = []
        data0 = self.input0
        data1 = self.input1
        collect_stats = self.ugsp_enabled and self.ugsp_stats_every > 0 and (
            self._ugsp_forward_count % self.ugsp_stats_every == 0
        )
        total_started = self._timing_start(data0.pos, collect_stats)

        # Preserve the original input-feature experiment: both first channels
        # are built from the original position/colour Change Mask.
        color0, color1 = data0.x[:, 1:], data1.x[:, 1:]
        nn_list00 = knn(data1.pos, data0.pos, 1, data1.batch, data0.batch)
        nn_list10 = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
        input0, original_prior0 = self.get_mask_v2(
            data1.pos, data0.pos, color1, color0, 3.0, 0.6, nn_list00
        )
        input1, original_prior1 = self.get_mask_v2(
            data0.pos, data1.pos, color0, color1, 3.0, 0.6, nn_list10
        )
        data0.x[:, 0] = input0.squeeze(-1)
        data1.x[:, 0] = input1.squeeze(-1)

        # When disabled, UGSP and its hierarchy are never evaluated; the same
        # trainable CADA modules instead receive the original prior.
        if self.ugsp_enabled:
            prior0 = self.ugsp(
                data0.pos,
                data1.pos,
                data0.batch,
                data1.batch,
                collect_timings=collect_stats,
            )
            self.M_GP = prior0
            self.M_GP_hard = self.ugsp.last_details["hard"]
        else:
            prior0 = original_prior0
            self.M_GP = None
            self.M_GP_hard = None
        prior1 = original_prior1
        self.M_G_layers = []
        self.M_G_reverse_layers = []
        projection_time = 0.0
        cada_time = 0.0

        started = self._timing_start(data0.pos, collect_stats)
        data0, prior0 = self._down_with_prior(data0, prior0, self.down_modules_1[0], self.pre_computed)
        data1, prior1 = self._down_with_prior(data1, prior1, self.down_modules_2[0], self.pre_computed_target)
        projection_time += self._timing_elapsed(data0.pos, started, collect_stats)
        self.M_G_layers.append(prior0)
        self.M_G_reverse_layers.append(prior1)
        encoder_levels.append((data0.clone(), data1.clone(), prior0, prior1))

        # The geometric-only pass produces the next encoder level. Once LMSP is
        # available, the decoder skip differences are rebuilt with joint priors.
        started = self._timing_start(data0.pos, collect_stats)
        data0.x = self.self_att_list[0](data0.x, None, False, prior0)
        data1.x = self.self_att_list[0](data1.x, None, False, prior1)
        cada_time += self._timing_elapsed(data0.pos, started, collect_stats)
        diff0, diff1 = self._cross_temporal_difference(data0, data1, data0.x, data1.x)
        data0.x = torch.cat((data0.x, diff0.x), axis=1)
        data1.x = torch.cat((data1.x, diff1.x), axis=1)

        for i in range(1, len(self.down_modules_1) - 1):
            started = self._timing_start(data0.pos, collect_stats)
            data0, prior0 = self._down_with_prior(data0, prior0, self.down_modules_1[i], self.pre_computed)
            data1, prior1 = self._down_with_prior(data1, prior1, self.down_modules_2[i], self.pre_computed_target)
            projection_time += self._timing_elapsed(data0.pos, started, collect_stats)
            self.M_G_layers.append(prior0)
            self.M_G_reverse_layers.append(prior1)
            encoder_levels.append((data0.clone(), data1.clone(), prior0, prior1))

            started = self._timing_start(data0.pos, collect_stats)
            data0.x = self.self_att_list[i](data0.x, None, False, prior0)
            data1.x = self.self_att_list[i](data1.x, None, False, prior1)
            cada_time += self._timing_elapsed(data0.pos, started, collect_stats)
            diff0, diff1 = self._cross_temporal_difference(data0, data1, data0.x, data1.x)
            data0.x = torch.cat((data0.x, diff0.x), axis=1)
            data1.x = torch.cat((data1.x, diff1.x), axis=1)

        # Deep encoder features define the statistical change descriptors.
        started = self._timing_start(data0.pos, collect_stats)
        data0, prior0 = self._down_with_prior(data0, prior0, self.down_modules_1[-1], self.pre_computed)
        data1, prior1 = self._down_with_prior(data1, prior1, self.down_modules_2[-1], self.pre_computed_target)
        projection_time += self._timing_elapsed(data0.pos, started, collect_stats)
        self.M_G_layers.append(prior0)
        self.M_G_reverse_layers.append(prior1)

        labels0_deep = self._project_labels(
            self.labels0, self.input0.pos, data0.pos, self.input0.batch, data0.batch
        )
        labels1_deep = self._project_labels(
            self.labels1, self.input1.pos, data1.pos, self.input1.batch, data1.batch
        )
        if self.lmsp_enabled:
            stat0, stat1 = self.lmsp(
                data0.x,
                data1.x,
                data0.pos,
                data1.pos,
                data0.batch,
                data1.batch,
                labels0_deep,
                labels1_deep,
                update=self.training,
            )
            lmsp_ready = self.lmsp.ready
        else:
            stat0, stat1 = None, None
            lmsp_ready = False

        self.M_stat = stat0
        self.M_stat_reverse = stat1
        self.M_stat_layers = []
        self.M_stat_reverse_layers = []
        self.M_joint_layers = []
        self.M_joint_reverse_layers = []

        # Propagate the deepest statistical mask back to every encoder scale,
        # fuse it with UAGP, and rebuild all decoder skip differences.
        for level_id, (level0, level1, geo0, geo1) in enumerate(encoder_levels):
            if lmsp_ready:
                stat_level0 = self._project_prior(stat0, data0, level0)
                stat_level1 = self._project_prior(stat1, data1, level1)
            else:
                stat_level0 = None
                stat_level1 = None
            joint0 = self._fuse_priors(geo0, stat_level0)
            joint1 = self._fuse_priors(geo1, stat_level1)
            self.M_stat_layers.append(stat_level0)
            self.M_stat_reverse_layers.append(stat_level1)
            self.M_joint_layers.append(joint0)
            self.M_joint_reverse_layers.append(joint1)

            started = self._timing_start(level0.pos, collect_stats)
            feature0 = self.self_att_list[level_id](level0.x, None, False, joint0)
            feature1 = self.self_att_list[level_id](level1.x, None, False, joint1)
            cada_time += self._timing_elapsed(level0.pos, started, collect_stats)
            skip0, skip1 = self._cross_temporal_difference(level0, level1, feature0, feature1)
            skip0.mask, skip1.mask = joint0, joint1
            stack_down.append([skip0, skip1])

        joint0 = self._fuse_priors(prior0, stat0 if lmsp_ready else None)
        joint1 = self._fuse_priors(prior1, stat1 if lmsp_ready else None)
        self.M_stat_layers.append(stat0 if lmsp_ready else None)
        self.M_stat_reverse_layers.append(stat1 if lmsp_ready else None)
        self.M_joint_layers.append(joint0)
        self.M_joint_reverse_layers.append(joint1)
        self.M_joint = joint0
        self.M_joint_reverse = joint1
        data0.mask, data1.mask = joint0, joint1

        started = self._timing_start(data0.pos, collect_stats)
        data0.x = self.self_att_list[-1](data0.x, None, False, joint0)
        data1.x = self.self_att_list[-1](data1.x, None, False, joint1)
        cada_time += self._timing_elapsed(data0.pos, started, collect_stats)
        diff0, diff1 = self._cross_temporal_difference(data0, data1, data0.x, data1.x)

        data0.x = torch.cat((data0.x, diff0.x), axis=1)
        data1.x = torch.cat((data1.x, diff1.x), axis=1)
        innermost = False
        if not isinstance(self.inner_modules[0], Identity):
            stack_down.append([diff0, diff1])
            data0 = self.inner_modules[0](data0)
            data1 = self.inner_modules[0](data1)
            innermost = True
        for i in range(len(self.up_modules_1)):
            if i == 0 and innermost:
                diff0, diff1 = stack_down.pop()
                data0 = self.up_modules_1[i]((data0, diff0))
                data1 = self.up_modules_2[i]((data1, diff1))
            else:
                diff0, diff1 = stack_down.pop()
                data0 = self.up_modules_1[i]((data0, diff0), precomputed=self.upsample_target)
                data1 = self.up_modules_2[i]((data1, diff1), precomputed=self.upsample_target)

        last_feature0 = data0.x
        last_feature1 = data1.x
        if self._use_category:
            self.output0 = self.FC_layer(last_feature0, self.category)
            self.output1 = self.FC_layer(last_feature1, self.category)
        else:
            self.output0 = self.FC_layer(last_feature0)
            self.output1 = self.FC_layer(last_feature1)

        self.aux_output0 = None
        self.aux_output1 = None
        if self.absh_enabled and self.training and self.labels0 is not None and self.labels1 is not None:
            if self._use_category:
                self.aux_output0 = self.FC_layer_aux(last_feature0, self.category)
                self.aux_output1 = self.FC_layer_aux(last_feature1, self.category)
            else:
                self.aux_output0 = self.FC_layer_aux(last_feature0)
                self.aux_output1 = self.FC_layer_aux(last_feature1)

        if self.labels0 is not None and self.labels1 is not None:
            self.compute_loss()

        self.data_visual0 = self.input0
        self.data_visual0.pred = torch.max(self.output0, -1)[1]
        self.data_visual1 = self.input1
        self.data_visual1.pred = torch.max(self.output1, -1)[1]

        self.output = [self.output0, self.output1]

        if collect_stats:
            total_time = self._timing_elapsed(data0.pos, total_started, True)
            self._log_ugsp_statistics(projection_time, cada_time, total_time)
        self._ugsp_forward_count += 1

        return self.output0, self.output1

    def _down_with_prior(self, data, prior, down_module, precomputed):
        """Run an encoder level and reuse its existing support mapping."""
        block_idx = int(getattr(data, "block_idx", 0))
        data.cada_prior = prior
        output = down_module(data, precomputed=precomputed)

        samplers = down_module.sampler
        has_sampling = (
            any(sampler is not None for sampler in samplers)
            if isinstance(samplers, list)
            else samplers is not None
        )
        if not has_sampling:
            projected = prior
        elif precomputed is None and getattr(output, "cada_prior", None) is not None:
            projected = output.cada_prior.reshape(-1, 1)
        else:
            if precomputed is not None:
                support_index = precomputed[block_idx].idx_neighboors
            else:
                support_index = output.idx_neighboors
            projected = self._aggregate_prior(prior, support_index, output.pos.shape[0])
        output.cada_prior = projected
        return output, projected

    @staticmethod
    def _aggregate_prior(prior, support_index, query_count):
        support_index = support_index.long()
        if support_index.dim() == 1:
            support_index = support_index.reshape(query_count, -1)
        valid = (support_index >= 0) & (support_index < prior.shape[0])
        safe_index = support_index.clamp(0, max(prior.shape[0] - 1, 0))
        values = prior[safe_index].squeeze(-1) * valid.to(prior.dtype)
        count = valid.to(prior.dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
        return values.sum(dim=1, keepdim=True) / count

    @staticmethod
    def _extract_class_counts(dataset, num_classes):
        candidates = []
        for name in ("train_data", "train_dataset"):
            try:
                value = getattr(dataset, name, None)
            except Exception:
                value = None
            if value is not None:
                candidates.append(value)
        candidates.append(dataset)

        for source in candidates:
            for name in ("absh_class_counts", "nb_elt_class"):
                counts = getattr(source, name, None)
                if counts is None:
                    continue
                counts = torch.as_tensor(counts, dtype=torch.float32).reshape(-1)
                if counts.numel() == num_classes and counts.sum() > 0:
                    return counts
        log.warning("ABSH could not find training-set class counts; using uniform retention")
        return torch.ones(num_classes, dtype=torch.float32)

    @staticmethod
    def _nearest_indices(support, query, support_batch, query_batch):
        edge = knn(support, query, 1, support_batch, query_batch)
        if edge.shape[1] != query.shape[0]:
            raise ValueError("Could not find one corresponding point for every query")
        nearest = torch.empty(query.shape[0], dtype=torch.long, device=query.device)
        nearest[edge[0]] = edge[1]
        return nearest

    @classmethod
    def _cross_temporal_difference(cls, data0, data1, feature0, feature1):
        match01 = cls._nearest_indices(data1.pos, data0.pos, data1.batch, data0.batch)
        match10 = cls._nearest_indices(data0.pos, data1.pos, data0.batch, data1.batch)
        diff0 = data0.clone()
        diff1 = data1.clone()
        diff0.x = feature0 - feature1[match01]
        diff1.x = feature1 - feature0[match10]
        return diff0, diff1

    @classmethod
    def _project_labels(cls, labels, support_pos, query_pos, support_batch, query_batch):
        if labels is None:
            return None
        nearest = cls._nearest_indices(support_pos, query_pos, support_batch, query_batch)
        return labels.reshape(-1)[nearest]

    @classmethod
    def _project_prior(cls, prior, support_data, query_data):
        nearest = cls._nearest_indices(
            support_data.pos,
            query_data.pos,
            support_data.batch,
            query_data.batch,
        )
        return prior[nearest]

    def _fuse_priors(self, geometric, statistical):
        if statistical is None:
            return geometric
        alpha = self.lmsp_fusion_alpha
        return alpha * geometric + (1.0 - alpha) * statistical

    @staticmethod
    def _timing_start(reference, enabled):
        if not enabled:
            return None
        if reference.is_cuda:
            torch.cuda.synchronize(reference.device)
        return time.perf_counter()

    @staticmethod
    def _timing_elapsed(reference, started, enabled):
        if not enabled or started is None:
            return 0.0
        if reference.is_cuda:
            torch.cuda.synchronize(reference.device)
        return time.perf_counter() - started

    def _log_ugsp_statistics(self, projection_time, cada_time, total_time):
        prior = self.M_GP.squeeze(-1)
        hard = self.M_GP_hard.squeeze(-1).bool()
        details = self.ugsp.last_details
        count0 = details["neighbor_count0"]
        count1 = details["neighbor_count1"]

        def tensor_stats(value):
            value = value.float()
            return "mean=%.4f std=%.4f min=%.4f max=%.4f" % (
                value.mean().item(),
                value.std(unbiased=False).item(),
                value.min().item(),
                value.max().item(),
            )

        def neighbor_stats(value):
            value = value.float()
            return "mean=%.2f median=%.2f min=%d max=%d p<3=%.3f p<5=%.3f p<10=%.3f" % (
                value.mean().item(),
                value.median().item(),
                int(value.min().item()),
                int(value.max().item()),
                (value < 3).float().mean().item(),
                (value < 5).float().mean().item(),
                (value < 10).float().mean().item(),
            )

        log.info("UGSP neighbors P0: %s", neighbor_stats(count0))
        log.info("UGSP neighbors P1: %s", neighbor_stats(count1))
        log.info(
            "UGSP prior: %s p>0.5=%.4f hard_positive=%.4f",
            tensor_stats(prior),
            (prior > 0.5).float().mean().item(),
            hard.float().mean().item(),
        )
        for key in ("d_l", "g_l", "d_o", "g_o", "ratio_l", "ratio_o"):
            log.info("UGSP %s: %s", key, tensor_stats(details[key]))

        if self.labels0 is not None and self.labels0.numel() == prior.numel():
            labels = self.labels0.reshape(-1)
            valid_label = torch.ones_like(labels, dtype=torch.bool)
            if self._ignore_label is not None:
                valid_label &= labels != self._ignore_label
            label_prior = prior[valid_label]
            label_hard = hard[valid_label]
            changed = labels[valid_label] != 0
            true_positive = (label_hard & changed).sum().float()
            precision = true_positive / label_hard.sum().clamp_min(1)
            recall = true_positive / changed.sum().clamp_min(1)
            f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
            changed_mean = label_prior[changed].mean().item() if changed.any() else float("nan")
            unchanged_mean = label_prior[~changed].mean().item() if (~changed).any() else float("nan")
            log.info(
                "UGSP labels: precision=%.4f recall=%.4f f1=%.4f changed_mean=%.4f unchanged_mean=%.4f",
                precision.item(), recall.item(), f1.item(), changed_mean, unchanged_mean,
            )
            for key in ("ratio_l", "ratio_o"):
                value = details[key][valid_label]
                positive_mean = value[changed].mean().item() if changed.any() else float("nan")
                negative_mean = value[~changed].mean().item() if (~changed).any() else float("nan")
                log.info("UGSP %s labels: changed=%.4f unchanged=%.4f", key, positive_mean, negative_mean)

        log.info(
            "UGSP timing(s): geometry=%.4f correspondence=%.4f hierarchy=%.4f CADA=%.4f total=%.4f",
            self.ugsp.last_timings["geometry"],
            self.ugsp.last_timings["correspondence"],
            projection_time,
            cada_time,
            total_time,
        )

    def sampler_data(self, data, samplers):
        for sampler in samplers:
            if sampler is not None:
                data = sampler(data.clone())
        return data

    def compute_loss(self):
        if self._weight_classes is not None:
            self._weight_classes = self._weight_classes.to(self.output0.device)
        self.loss = 0

        # Get regularization on weights
        if self.lambda_reg:
            self.loss_reg = self.get_regularization_loss(regularizer_type="l2", lambda_reg=self.lambda_reg)
            self.loss += self.loss_reg

        # Collect internal losses and set them with self and them to self for later tracking
        if self.lambda_internal_losses:
            print('lambda_internal_losses')
            self.loss += self.collect_internal_losses(lambda_weight=self.lambda_internal_losses)

        # The original head keeps full supervision on both temporal point sets.
        if self._ignore_label is not None:
            self.loss_main = F.nll_loss(
                self.output0, self.labels0, weight=self._weight_classes, ignore_index=self._ignore_label
            ) + F.nll_loss(
                self.output1, self.labels1, weight=self._weight_classes, ignore_index=self._ignore_label
            )
        else:
            self.loss_main = F.nll_loss(
                self.output0, self.labels0, weight=self._weight_classes
            ) + F.nll_loss(self.output1, self.labels1, weight=self._weight_classes)

        if self.absh_enabled and self.aux_output0 is not None and self.aux_output1 is not None:
            self.loss_bal = self.absh_loss(
                (self.aux_output0, self.aux_output1),
                (self.labels0, self.labels1),
            )
        else:
            self.loss_bal = self.loss_main.new_zeros(())

        self.loss_cd = self.loss_main + self.lambda_bal * self.loss_bal
        self.loss_seg = self.loss_cd
        if torch.isnan(self.loss_cd):
            raise FloatingPointError("Change-detection loss became NaN")
        self.loss += self.loss_cd

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss.backward()  # calculate gradients of network G w.r.t. loss_G

    def get_mask_v2(self, pos_1, pos_2, f_1, f_2, pos_th, f_th, knearest_idx_2):
        knearest_idx_2 = knearest_idx_2.reshape(knearest_idx_2.shape[0], -1, 1)
        # 几何差异
        nearest_pos_2of1 = pos_1[knearest_idx_2[1, :, :], :]
        pos_2_ex = pos_2.unsqueeze(1).repeat(1, nearest_pos_2of1.shape[1], 1)
        dis_mat_2 = torch.mean(torch.sqrt(torch.sum(torch.square(pos_2_ex - nearest_pos_2of1), dim=-1)), dim=-1).view(-1, 1)
        pos_m_1 = torch.where(dis_mat_2 > pos_th, 1.0, dis_mat_2 / pos_th)
        # 纹理差异
        nearest_f_2of1 = f_1[knearest_idx_2[1, :, :], :]
        f_2_ex = f_2.unsqueeze(1).repeat(1, nearest_f_2of1.shape[1], 1)
        f_mat_2 = torch.abs(torch.mean(f_2_ex - nearest_f_2of1, dim=-1).view(-1, 1))
        f_m_1 = torch.where(f_mat_2 > f_th, 1.0, f_mat_2 / f_th)

        input = 0.5 * pos_m_1 + 0.5 * f_m_1
        mask = input
        return input, mask

class Attention(nn.Module):
    def __init__(self, channels, gate_floor=0.1):
        super(Attention, self).__init__()
        if not 0.0 <= gate_floor < 1.0:
            raise ValueError("CADA gate_floor must be in [0, 1)")
        self.gate_floor = float(gate_floor)
        self.energy_conv = Linear(channels, channels)
        self.v_conv = Linear(channels, channels)
        self.pos_norm = FastBatchNorm1d(channels)

        self.trans_conv = Linear(channels, channels)
        self.after_norm = FastBatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, v, xyz, add_pos, mask):
        # b, n, c
        if add_pos:
            value = self.pos_norm.to(v.device)(v + xyz)
        else:
            value = v

        # b, c, n
        energy = self.energy_conv.to(value.device)(value)
        value = self.v_conv.to(value.device)(value)

        if mask is not None:
            energy = torch.multiply(energy, mask)

        attention = self.softmax(energy)
        x_r = torch.multiply(value, attention)
        response = self.act.to(value.device)(
            self.after_norm.to(value.device)(self.trans_conv.to(value.device)(x_r))
        )
        if mask is not None:
            gate = self.gate_floor + (1.0 - self.gate_floor) * mask
            response = torch.multiply(response, gate)
        x = response + v
        return x
