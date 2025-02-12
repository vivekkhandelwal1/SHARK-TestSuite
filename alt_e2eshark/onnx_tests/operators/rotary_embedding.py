# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..helper_classes import AzureDownloadableModel, BuildAModel
from onnx.helper import make_tensor, make_tensor_value_info
from onnx import TensorProto
from e2e_testing.registry import register_test

import numpy
from onnx import TensorProto
import onnx
from onnx.helper import make_tensor_value_info


class RotaryEmbeddingWithNoAttributesModel(BuildAModel):

    def construct_initializers(self):
        self.initializers = [
            make_tensor(
                "i1",
                TensorProto.FLOAT,
                [1, 3, 2, 6],
                [
                    -1.0408,
                    0.9166,
                    -1.3042,
                    -1.1097,
                    -1.2188,
                    1.1676,
                    -1.0574,
                    -0.1188,
                    -0.9078,
                    0.3452,
                    -0.5713,
                    -0.2351,
                    1.0076,
                    -0.7529,
                    -0.225,
                    -0.4327,
                    -1.5071,
                    -0.4586,
                    -0.848,
                    0.5266,
                    -1.2944,
                    -0.0243,
                    -0.2354,
                    -0.7087,
                    -0.8663,
                    -0.2656,
                    0.1665,
                    0.7911,
                    -0.932,
                    -0.8579,
                    -0.9647,
                    -0.0991,
                    -0.2994,
                    -0.065,
                    -1.572,
                    -1.3211,
                ],
            )
        ]  # input
        self.initializers += [
            make_tensor("i2", TensorProto.INT64, [1, 2], [0, 1])
        ]  # position_ids
        self.initializers += [
            make_tensor(
                "i3",
                TensorProto.FLOAT,
                [4, 3],
                [
                    1.0,
                    1.0,
                    1.0,
                    0.5403,
                    0.9989,
                    1.0,
                    -0.4161,
                    0.9957,
                    1.0,
                    -0.99,
                    0.9903,
                    1.0,
                ],
            )
        ]  # cos_cache
        self.initializers += [
            make_tensor(
                "i4",
                TensorProto.FLOAT,
                [4, 3],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.8415,
                    0.0464,
                    0.0022,
                    0.9093,
                    0.0927,
                    0.0043,
                    0.1411,
                    0.1388,
                    0.0065,
                ],
            )
        ]  # sin_cache

    def construct_nodes(self):
        app_node = self.get_app_node()
        app_node(
            "RotaryEmbedding", ["i1", "i2", "i3", "i4"], ["o1"]
        )  # Order of args: input, position_ids, cos_cache, sin_cache

    def construct_i_o_value_info(self):
        o1 = make_tensor_value_info("o1", TensorProto.FLOAT, [1, 3, 2, 6])
        self.output_vi = [o1]


register_test(RotaryEmbeddingWithNoAttributesModel, "rotary_emb_no_attrs")


class RotaryEmbeddingWithInterleavedModel(BuildAModel):

    def construct_initializers(self):
        self.initializers = [
            make_tensor(
                "i1",
                TensorProto.FLOAT,
                [1, 2, 3, 4],
                [
                    -1.0408,
                    0.9166,
                    -1.3042,
                    -1.1097,
                    -1.2188,
                    1.1676,
                    -1.0574,
                    -0.1188,
                    -0.811,
                    0.6737,
                    -1.1233,
                    -0.0919,
                    -0.132,
                    -0.2751,
                    -0.235,
                    0.0937,
                    -0.7396,
                    -1.2425,
                    -0.1752,
                    0.699,
                    -0.6861,
                    0.7202,
                    0.1963,
                    0.6142,
                ],
            )
        ]  # input
        self.initializers += [
            make_tensor("i2", TensorProto.INT64, [1], [0])
        ]  # position_ids
        self.initializers += [
            make_tensor(
                "i3",
                TensorProto.FLOAT,
                [8, 2],
                [
                    1.0,
                    1.0,
                    0.5403,
                    0.9999,
                    -0.4161,
                    0.9998,
                    -0.99,
                    0.9996,
                    -0.6536,
                    0.9992,
                    0.2837,
                    0.9988,
                    0.9602,
                    0.9982,
                    0.7539,
                    0.9976,
                ],
            )
        ]  # cos_cache
        self.initializers += [
            make_tensor(
                "i4",
                TensorProto.FLOAT,
                [8, 2],
                [
                    0.0,
                    0.0,
                    0.8415,
                    0.01,
                    0.9093,
                    0.02,
                    0.1411,
                    0.03,
                    -0.7568,
                    0.04,
                    -0.9589,
                    0.05,
                    -0.2794,
                    0.06,
                    0.657,
                    0.0699,
                ],
            )
        ]  # sin_cache

    def construct_nodes(self):
        app_node = self.get_app_node()
        app_node(
            "RotaryEmbedding", ["i1", "i2", "i3", "i4"], ["o1"], interleaved=1
        )  # Order of args: input, position_ids, cos_cache, sin_cache

    def construct_i_o_value_info(self):
        o1 = make_tensor_value_info("o1", TensorProto.FLOAT, [1, 2, 3, 4])
        self.output_vi = [o1]


register_test(RotaryEmbeddingWithInterleavedModel, "rotary_emb_with_interleaved")


class RotaryEmbeddingWithRotaryDimModel(BuildAModel):

    def construct_initializers(self):
        self.initializers = [
            make_tensor(
                "i1",
                TensorProto.FLOAT,
                [1, 2, 6],
                [
                    -1.0408,
                    0.9166,
                    -1.3042,
                    -1.1097,
                    -1.2188,
                    1.1676,
                    1.0076,
                    -0.7529,
                    -0.225,
                    -0.4327,
                    -1.5071,
                    -0.4586,
                ],
            )
        ]  # input
        self.initializers += [
            make_tensor("i2", TensorProto.INT64, [1, 2], [0, 1])
        ]  # position_ids
        self.initializers += [
            make_tensor(
                "i3",
                TensorProto.FLOAT,
                [2, 2],
                [1.0, 1.0, 1.0, 0.5403],
            )
        ]  # cos_cache
        self.initializers += [
            make_tensor(
                "i4",
                TensorProto.FLOAT,
                [2, 2],
                [0.0, 0.0, 0.0, 0.8415],
            )
        ]  # sin_cache

    def construct_nodes(self):
        app_node = self.get_app_node()
        app_node(
            "RotaryEmbedding",
            ["i1", "i2", "i3", "i4"],
            ["o1"],
            num_heads=1,
            rotary_embedding_dim=4,
        )  # Order of args: input, position_ids, cos_cache, sin_cache

    def construct_i_o_value_info(self):
        o1 = make_tensor_value_info("o1", TensorProto.FLOAT, [1, 2, 3, 4])
        self.output_vi = [o1]


register_test(RotaryEmbeddingWithRotaryDimModel, "rotary_emb_with_rotary_dim")
