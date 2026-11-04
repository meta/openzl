# Copyright (c) Meta Platforms, Inc. and affiliates.


import sys
from typing import List, Tuple
from unittest import TestCase

import numpy as np
from openzl import ext


class TestOpenzlSys(TestCase):
    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

    def _round_trip(self, compressor: ext.Compressor, inputs: List[ext.Input]) -> bytes:
        cctx = ext.CCtx()
        cctx.ref_compressor(compressor)
        compressed = cctx.compress(inputs)
        decompressed = ext.DCtx().decompress(compressed)
        assert len(decompressed) == len(inputs)
        for inp, out in zip(inputs, decompressed):
            assert inp.type == out.type
            assert np.array_equal(inp.content.as_nparray(), out.content.as_nparray())
        return compressed

    def to_arrays(self, data: List[str]) -> Tuple[np.array, np.array]:
        content = np.frombuffer(b"".join([x.encode() for x in data]), dtype=np.uint8)
        lengths = np.array([len(x) for x in data], dtype=np.uint32)
        return content, lengths

    def test_round_trip(self) -> None:
        compressor = ext.Compressor()
        compressor.set_parameter(ext.CParam.FormatVersion, 15)
        graph = compressor.get_graph("zl.zstd")
        assert graph is not None
        compressor.select_starting_graph(graph)
        cctx = ext.CCtx()
        cctx.ref_compressor(compressor)

        data = b"hello world hello world hello hello world world hello world hello helloworldhellohelloworld"
        inputs = [ext.Input(ext.Type.Serial, np.frombuffer(data, dtype=np.uint8))]
        self._round_trip(compressor, inputs)

    def test_custom_codec(self) -> None:
        desc = ext.MultiInputCodecDescription(
            id=1,
            name="plus_one",
            input_types=[ext.Type.Numeric],
            singleton_output_types=[ext.Type.Numeric],
        )

        class PlusOneEncoder(ext.CustomEncoder):
            def multi_input_description(self) -> None:
                return desc

            def encode(self, state: ext.EncoderState) -> None:
                input = state.inputs[0].content.as_nparray()
                output = state.create_output(0, len(input), input.dtype.itemsize)
                output.mut_content.as_nparray()[:] = input + 1
                output.commit(len(input))

        class PlusOneDecoder(ext.CustomDecoder):
            def multi_input_description(self) -> None:
                return desc

            def decode(self, state: ext.DecoderState) -> None:
                input = state.singleton_inputs[0].content.as_nparray()
                output = state.create_output(0, len(input), input.dtype.itemsize)
                output.mut_content.as_nparray()[:] = input - 1
                output.commit(len(input))

        compressor = ext.Compressor()
        compressor.set_parameter(ext.CParam.FormatVersion, 15)
        node = compressor.register_custom_encoder(PlusOneEncoder())
        graph = compressor.get_graph("zl.zstd")
        graph = compressor.build_static_graph(node, [graph])

        cctx = ext.CCtx()
        cctx.ref_compressor(compressor)

        data = np.array([0] * 1000 + [1, 2, 3], dtype=np.uint32)
        compressed = cctx.compress([ext.Input(ext.Type.Numeric, data)])

        dctx = ext.DCtx()
        dctx.register_custom_decoder(PlusOneDecoder())
        decompressed = dctx.decompress(compressed)
        assert len(decompressed) == 1
        assert decompressed[0].type == ext.Type.Numeric
        round_tripped = decompressed[0].content.as_nparray()
        assert np.array_equal(data, round_tripped)

        self.assertRaises(Exception, lambda: ext.DCtx().decompress(compressed))

    def test_custom_decoder_exception_is_useful(self) -> None:
        desc = ext.MultiInputCodecDescription(
            id=1,
            name="plus_one",
            input_types=[ext.Type.Numeric],
            singleton_output_types=[ext.Type.Numeric],
        )

        class PlusOneEncoder(ext.CustomEncoder):
            def multi_input_description(self) -> None:
                return desc

            def encode(self, state: ext.EncoderState) -> None:
                input = state.inputs[0].content.as_nparray()
                output = state.create_output(0, len(input), input.dtype.itemsize)
                output.mut_content.as_nparray()[:] = input + 1
                output.commit(len(input))

        class PlusOneDecoder(ext.CustomDecoder):
            def multi_input_description(self) -> None:
                return desc

            def decode(self, state: ext.DecoderState) -> None:
                raise Exception("xxxxxxxxxx")

        compressor = ext.Compressor()
        compressor.set_parameter(ext.CParam.FormatVersion, 15)
        node = compressor.register_custom_encoder(PlusOneEncoder())
        graph = compressor.get_graph("zl.zstd")
        graph = compressor.build_static_graph(node, [graph])

        cctx = ext.CCtx()
        cctx.ref_compressor(compressor)

        data = np.array([0] * 1000 + [1, 2, 3], dtype=np.uint32)
        compressed = cctx.compress([ext.Input(ext.Type.Numeric, data)])

        dctx = ext.DCtx()
        dctx.register_custom_decoder(PlusOneDecoder())
        self.assertRaisesRegex(
            Exception, ".*xxxxxxxxxx.*", lambda: dctx.decompress(compressed)
        )

    def test_custom_encoder_exception_is_useful(self) -> None:
        desc = ext.MultiInputCodecDescription(
            id=1,
            name="plus_one",
            input_types=[ext.Type.Numeric],
            singleton_output_types=[ext.Type.Numeric],
        )

        class PlusOneEncoder(ext.CustomEncoder):
            def multi_input_description(self) -> None:
                return desc

            def encode(self, state: ext.EncoderState) -> None:
                raise Exception("xxxxxxxxxx")

        compressor = ext.Compressor()
        compressor.set_parameter(ext.CParam.FormatVersion, 15)
        node = compressor.register_custom_encoder(PlusOneEncoder())
        graph = compressor.get_graph("zl.zstd")
        graph = compressor.build_static_graph(node, [graph])

        cctx = ext.CCtx()
        cctx.ref_compressor(compressor)

        data = np.array([0] * 1000 + [1, 2, 3], dtype=np.uint32)
        self.assertRaisesRegex(
            Exception,
            ".*xxxxxxxxxx.*",
            lambda: cctx.compress([ext.Input(ext.Type.Numeric, data)]),
        )

    def test_custom_codec_with_strings(self) -> None:
        desc = ext.MultiInputCodecDescription(
            id=1,
            name="reverse_string",
            input_types=[ext.Type.String],
            singleton_output_types=[ext.Type.String],
        )

        class ReverseStringEncoder(ext.CustomEncoder):
            def multi_input_description(self) -> None:
                return desc

            def encode(self, state: ext.EncoderState) -> None:
                content = state.inputs[0].content.as_nparray()
                lengths = state.inputs[0].string_lens.as_nparray()
                output = state.create_output(0, len(content), 1)
                output.reserve_string_lens(len(lengths))

                out_content = output.mut_content.as_nparray()
                out_lengths = output.mut_string_lens.as_nparray()

                offset = 0
                for length in lengths:
                    begin = offset
                    end = offset + length
                    out_content[begin:end] = content[begin:end][::-1]
                    offset += length

                out_lengths[:] = lengths
                output.commit(len(lengths))

        class ReverseStringDecoder(ext.CustomDecoder):
            def multi_input_description(self) -> None:
                return desc

            def decode(self, state: ext.DecoderState) -> None:
                content = state.singleton_inputs[0].content.as_nparray()
                lengths = state.singleton_inputs[0].string_lens.as_nparray()
                output = state.create_output(0, len(content), 1)
                output.reserve_string_lens(len(lengths))

                out_content = output.mut_content.as_nparray()
                out_lengths = output.mut_string_lens.as_nparray()

                offset = 0
                for length in lengths:
                    begin = offset
                    end = offset + length
                    out_content[begin:end] = content[begin:end][::-1]
                    offset += length

                out_lengths[:] = lengths
                output.commit(len(lengths))

        compressor = ext.Compressor()
        compressor.set_parameter(ext.CParam.FormatVersion, 15)
        node = compressor.register_custom_encoder(ReverseStringEncoder())
        graph = compressor.get_graph("zl.store")
        graph = compressor.build_static_graph(node, [graph])

        cctx = ext.CCtx()
        cctx.ref_compressor(compressor)

        content, lengths = self.to_arrays(
            ["hello", "world", "abcde", "", "1", "12", "123"]
        )
        compressed = cctx.compress([ext.Input(ext.Type.String, content, lengths)])

        dctx = ext.DCtx()
        dctx.register_custom_decoder(ReverseStringDecoder())
        decompressed = dctx.decompress(compressed)
        assert len(decompressed) == 1
        assert decompressed[0].type == ext.Type.String
        rt_content = decompressed[0].content.as_nparray()
        rt_lengths = decompressed[0].string_lens.as_nparray()
        assert np.array_equal(content, rt_content)
        assert np.array_equal(lengths, rt_lengths)

    def test_function_graph_basic(self) -> None:
        compressor = ext.Compressor()
        compressor.set_parameter(ext.CParam.FormatVersion, 15)
        zstd = compressor.get_graph("zl.zstd")
        desc = ext.FunctionGraphDescription(
            name="test_graph",
            input_type_masks=[ext.TypeMask.Serial | ext.TypeMask.Numeric],
            custom_graphs=[zstd],
        )

        class BasicFunctionGraph(ext.FunctionGraph):
            def function_graph_description(self) -> ext.FunctionGraphDescription:
                return desc

            def graph(self, state: ext.GraphState) -> None:
                state.edges[0].set_destination(state.custom_graphs[0])

        compressor.register_function_graph(BasicFunctionGraph())

        data = b"x" * 1000 + b"hello_world_hello_world-hello"
        inputs = [ext.Input(ext.Type.Serial, np.frombuffer(data, dtype=np.uint8))]

        self._round_trip(compressor, inputs)

    def test_selector_basic(self) -> None:
        compressor = ext.Compressor()
        compressor.set_parameter(ext.CParam.FormatVersion, 15)
        zstd = compressor.get_graph("zl.zstd")

        is_called = False

        class BasicSelector(ext.Selector):
            def selector_description(self) -> ext.SelectorDescription:
                return ext.SelectorDescription(
                    name="BasicSelector",
                    input_type_mask=ext.TypeMask.Serial,
                    custom_graphs=[zstd],
                    local_params=ext.LocalParams({0: 0}),
                )

            def select(self, state: ext.SelectorState, input: ext.Input) -> ext.GraphID:
                assert input.content.as_bytes()[:1] == b"x"
                nonlocal is_called
                is_called = True
                idx = state.get_local_int_param(0)
                assert idx is not None
                return state.custom_graphs[idx]

        compressor.register_selector_graph(BasicSelector())

        data = b"x" * 1000 + b"hello_world_hello_world-hello"
        inputs = [ext.Input(ext.Type.Serial, np.frombuffer(data, dtype=np.uint8))]
        self._round_trip(compressor, inputs)
        assert is_called

    def test_bitpack(self) -> None:
        compressor = ext.Compressor()
        compressor.set_parameter(ext.CParam.FormatVersion, 15)
        graph = ext.graphs.Bitpack()(compressor)
        compressor.select_starting_graph(graph)

        data = np.array([7] * 10000, dtype=np.uint64)
        inputs = [ext.Input(ext.Type.Numeric, data)]
        self._round_trip(compressor, inputs)

    def test_run_bitpack(self) -> None:
        compressor = ext.Compressor()
        compressor.set_parameter(ext.CParam.FormatVersion, 15)

        class BitpackFunctionGraph(ext.FunctionGraph):
            def function_graph_description(self) -> ext.FunctionGraphDescription:
                return ext.FunctionGraphDescription(
                    name="bitpack_function_graph",
                    input_type_masks=[ext.TypeMask.Numeric],
                )

            def graph(self, state: ext.GraphState) -> None:
                ext.graphs.Bitpack().set_destination(state.edges[0])

        graph = compressor.register_function_graph(BitpackFunctionGraph())
        compressor.select_starting_graph(graph)

        data = np.array([7] * 10000, dtype=np.uint64)
        inputs = [ext.Input(ext.Type.Numeric, data)]
        self._round_trip(compressor, inputs)

    def test_brute_force(self) -> None:
        compressor = ext.Compressor()
        compressor.set_parameter(ext.CParam.FormatVersion, ext.MAX_FORMAT_VERSION)
        brute_force = ext.graphs.BruteForce(
            successors=[
                ext.graphs.Zstd()(compressor),
                ext.graphs.Entropy()(compressor),
                ext.graphs.Store()(compressor),
            ]
        )
        compressor.select_starting_graph(brute_force(compressor))

        data = np.array([7] * 10000, dtype=np.uint32)
        inputs = [ext.Input(ext.Type.Numeric, data)]
        compressed = self._round_trip(compressor, inputs)

        # BruteForce parameterizes a standard graph, so it is serializable
        serialized = compressor.serialize()
        deserialized = ext.Compressor()
        deps = deserialized.get_unmet_dependencies(serialized)
        assert len(deps.graph_names) == 0
        assert len(deps.node_names) == 0
        deserialized.deserialize(serialized)
        self.assertEqual(compressed, self._round_trip(deserialized, inputs))

    def test_brute_force_in_function_graph(self) -> None:
        compressor = ext.Compressor()
        compressor.set_parameter(ext.CParam.FormatVersion, ext.MAX_FORMAT_VERSION)

        class BruteForceFunctionGraph(ext.FunctionGraph):
            def function_graph_description(self) -> ext.FunctionGraphDescription:
                return ext.FunctionGraphDescription(
                    name="brute_force_function_graph",
                    input_type_masks=[ext.TypeMask.Numeric],
                )

            def graph(self, state: ext.GraphState) -> None:
                ext.graphs.BruteForce(
                    successors=[
                        ext.graphs.Entropy().base_graph,
                        ext.graphs.Store().base_graph,
                    ]
                ).set_destination(state.edges[0])

        graph = compressor.register_function_graph(BruteForceFunctionGraph())
        compressor.select_starting_graph(graph)

        data = np.array([7] * 10000, dtype=np.uint32)
        inputs = [ext.Input(ext.Type.Numeric, data)]
        self._round_trip(compressor, inputs)

    def test_quantize_offsets(self) -> None:
        compressor = ext.Compressor()
        compressor.set_parameter(ext.CParam.FormatVersion, 15)
        graph = ext.nodes.QuantizeOffsets()(
            compressor,
            codes=ext.graphs.Entropy(),
            extra_bits=ext.graphs.Store(),
        )
        compressor.select_starting_graph(graph)

        data = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 10000, 1**30] + [1] * 10,
            dtype=np.uint32,
        )
        inputs = [ext.Input(ext.Type.Numeric, data)]
        self._round_trip(compressor, inputs)

        # Doesn't support 0
        data = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 10000, 1**30] + [1] * 10,
            dtype=np.uint32,
        )
        cctx = ext.CCtx()
        cctx.ref_compressor(compressor)
        self.assertRaises(
            Exception, lambda: cctx.compress([ext.Input(ext.Type.Numeric, data)])
        )

    def test_quantize_lengths(self) -> None:
        compressor = ext.Compressor()
        compressor.set_parameter(ext.CParam.FormatVersion, 15)
        graph = ext.nodes.QuantizeLengths()(
            compressor,
            codes=ext.graphs.Entropy(),
            extra_bits=ext.graphs.Store()(compressor),
        )
        compressor.select_starting_graph(graph)

        data = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 10000, 1**30] + [1] * 10,
            dtype=np.uint32,
        )
        inputs = [ext.Input(ext.Type.Numeric, data)]
        self._round_trip(compressor, inputs)

    def test_sddl(self) -> None:
        compressor = ext.Compressor()
        compressor.set_parameter(ext.CParam.FormatVersion, 15)
        graph = ext.graphs.SDDL(
            description=": Byte[_rem];", successor=ext.graphs.Store()(compressor)
        )(compressor)
        compressor.select_starting_graph(graph)

        data = b"hello world hello world hello hello world world hello world hello helloworldhellohelloworld"
        inputs = [ext.Input(ext.Type.Serial, data)]
        self._round_trip(compressor, inputs)

        self.assertRaises(
            Exception, lambda: ext.graphs.sddl(compressor, "???", ext.graphs.store())
        )

    def test_serialization(self) -> None:
        compressor1 = ext.Compressor()
        graph = ext.graphs.Constant()(compressor1)
        compressor1.select_starting_graph(graph)
        compressor1.set_parameter(ext.CParam.FormatVersion, ext.MAX_FORMAT_VERSION)

        compressor2 = ext.Compressor()
        deps = compressor2.get_unmet_dependencies(compressor1.serialize())
        assert len(deps.graph_names) == 0
        assert len(deps.node_names) == 0
        compressor2.deserialize(compressor1.serialize())

        data = np.array([42] * 1000, dtype=np.uint32)
        inputs = [ext.Input(ext.Type.Numeric, data)]
        compressed1 = self._round_trip(compressor1, inputs)
        compressed2 = self._round_trip(compressor2, inputs)

        self.assertEqual(compressed1, compressed2)

    def test_decompress_array_del_dctx(self) -> None:
        data = np.array([42] * 1000, dtype=np.uint32)
        compressor = ext.Compressor()
        compressor.set_parameter(ext.CParam.FormatVersion, ext.MAX_FORMAT_VERSION)
        graph = ext.graphs.Constant()(compressor)
        compressor.select_starting_graph(graph)
        compressed = self._round_trip(compressor, [ext.Input(ext.Type.Numeric, data)])

        dctx = ext.DCtx()
        decompressed = dctx.decompress(compressed)
        round_tripped = decompressed[0].content.as_nparray()
        del dctx
        del decompressed

        self.assertTrue(np.array_equal(data, round_tripped))

    def test_decompress_buffer(self) -> None:
        data = np.array([42] * 1000, dtype=np.uint32)
        compressor = ext.Compressor()
        compressor.set_parameter(ext.CParam.FormatVersion, ext.MAX_FORMAT_VERSION)
        graph = ext.graphs.Constant()(compressor)
        compressor.select_starting_graph(graph)
        compressed = self._round_trip(compressor, [ext.Input(ext.Type.Numeric, data)])

        class MyBuffer:
            def __init__(self, data):
                self.data = data

            def __buffer__(self, flag):
                return self.data.__buffer__(flag)

        buffers = [
            bytearray(compressed),
            np.frombuffer(compressed, dtype=np.uint8),
            MyBuffer(compressed),
        ]
        if sys.version_info < (3, 12):
            buffers.pop()
        for buffer in buffers:
            dctx = ext.DCtx()
            decompressed = dctx.decompress(buffer)
            round_tripped = decompressed[0].content.as_nparray()
            self.assertTrue(np.array_equal(data, round_tripped))

    def test_decompress_does_not_convert_arrays(self) -> None:
        buffer = np.frombuffer(b"Pan-Galactic Gargle Blaster\0", dtype=np.uint16)
        dctx = ext.DCtx()
        self.assertRaises(
            TypeError, "incompatible function arguments", dctx.decompress, buffer
        )
