// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/ext.hpp"

#include <nanobind/ndarray.h>

#include <utility>

#include "openzl/common/stream.h"
#include "openzl/openzl.hpp"
#include "openzl/zl_compressor.h"
#include "openzl/zl_version.h"

#include "openzl/ext/graphs.hpp"
#include "openzl/ext/nodes.hpp"

// Needs to be included exactly once in the extension
#include <nanobind/intrusive/counter.inl>

namespace openzl {
namespace py {
nb::ref<const PyInput> toPyInput(const Input& input)
{
    return new PyInput(std::move(const_cast<Input&>(input)));
}

std::vector<nb::ref<const PyInput>> toPyInputs(
        poly::span<const InputRef> inputs)
{
    std::vector<nb::ref<const PyInput>> pyInputs;
    pyInputs.reserve(inputs.size());
    for (auto& input : inputs) {
        pyInputs.push_back(toPyInput(input));
    }
    return pyInputs;
}

namespace {
using ConstBytes = nb::ndarray<
        uint8_t,
        nb::numpy,
        nb::ro,
        nb::shape<-1>,
        nb::device::cpu,
        nb::c_contig>;

void registerVersionInfo(nb::module_& m)
{
    m.attr("MIN_FORMAT_VERSION") = ZL_MIN_FORMAT_VERSION;
    m.attr("MAX_FORMAT_VERSION") = ZL_MAX_FORMAT_VERSION;

    m.attr("LIBRARY_VERSION_MAJOR")  = ZL_LIBRARY_VERSION_MAJOR;
    m.attr("LIBRARY_VERSION_MINOR")  = ZL_LIBRARY_VERSION_MINOR;
    m.attr("LIBRARY_VERSION_PATCH")  = ZL_LIBRARY_VERSION_PATCH;
    m.attr("LIBRARY_VERSION_NUMBER") = ZL_LIBRARY_VERSION_NUMBER;
}

void registerNodeID(nb::module_& m)
{
    nb::class_<NodeID>(m, "NodeID");
}

void registerGraphID(nb::module_& m)
{
    nb::class_<GraphID>(m, "GraphID");
}

void registerType(nb::module_& m)
{
    nb::enum_<Type>(m, "Type")
            .value("Serial", Type::Serial)
            .value("Struct", Type::Struct)
            .value("Numeric", Type::Numeric)
            .value("String", Type::String);
}

void registerTypeMask(nb::module_& m)
{
    nb::enum_<TypeMask>(m, "TypeMask", nb::is_flag())
            .value("Serial", TypeMask::Serial)
            .value("Struct", TypeMask::Struct)
            .value("Numeric", TypeMask::Numeric)
            .value("String", TypeMask::String)
            .value("None_", TypeMask::None)
            .value("Any", TypeMask::Any);
}

void registerCParam(nb::module_& m)
{
    nb::enum_<CParam>(m, "CParam")
            .value("StickyParameters", CParam::StickyParameters)
            .value("CompressionLevel", CParam::CompressionLevel)
            .value("DecompressionLevel", CParam::DecompressionLevel)
            .value("FormatVersion", CParam::FormatVersion)
            .value("PermissiveCompression", CParam::PermissiveCompression)
            .value("CompressedChecksum", CParam::CompressedChecksum)
            .value("ContentChecksum", CParam::ContentChecksum)
            .value("MinStreamSize", CParam::MinStreamSize);
}

void registerDParam(nb::module_& m)
{
    nb::enum_<DParam>(m, "DParam")
            .value("StickyParameters", DParam::StickyParameters)
            .value("CheckCompressedChecksum", DParam::CheckCompressedChecksum)
            .value("CheckContentChecksum", DParam::CheckContentChecksum);
}

template <typename... Args>
using NDArray = nb::ndarray<Args..., nb::device::cpu, nb::c_contig>;

template <typename... Args>
using LengthsNDArray = NDArray<Args..., uint32_t, nb::ndim<1>>;

template <typename... Args>
using InputNDArray = NDArray<Args..., nb::ro>;

template <typename... Args>
using InputLengthsNDArray = InputNDArray<Args..., uint32_t, nb::ndim<1>>;

Input inputFromArrays(
        Type type,
        const nb::ndarray<nb::ro, nb::device::cpu, nb::c_contig>& data,
        const poly::optional<nb::ndarray<
                nb::ro,
                nb::device::cpu,
                nb::c_contig,
                nb::ndim<1>>>& lengths)
{
    if (lengths.has_value() && type != Type::String) {
        throw Exception("Input: Passed lengths to non-string type");
    }
    if (data.ndim() != 1 && type != Type::Struct) {
        throw Exception(
                "Input: Non-struct types takes a 1-dimensional byte array");
    }
    if (data.dtype() != nb::dtype<uint8_t>() && type != Type::Numeric) {
        throw Exception("Input: Passed non-bytes data to non-numeric type");
    }

    switch (type) {
        case Type::Serial:
            return Input::refSerial(data.data(), data.nbytes());
        case Type::Struct:
            if (data.ndim() != 2) {
                throw Exception(
                        "Input: Struct takes a 2-dimensional byte array");
            }
            return Input::refStruct(data.data(), data.shape(0), data.shape(1));
        case Type::Numeric: {
            auto bits = data.dtype().bits;
            if (bits == 8) {
                return Input::refNumeric(data.data(), 1, data.size());
            } else if (bits == 16) {
                return Input::refNumeric(data.data(), 2, data.size());
            } else if (bits == 32) {
                return Input::refNumeric(data.data(), 4, data.size());
            } else if (bits == 64) {
                return Input::refNumeric(data.data(), 8, data.size());
            } else {
                throw Exception(
                        "Input: Numeric input takes 8-, 16-, 32-, or 64-bit data");
            }
        }
        case Type::String:
            if (!lengths.has_value()) {
                throw Exception("Input: Lengths not passed to string type");
            }
            if (lengths->dtype() != nb::dtype<uint32_t>()) {
                throw Exception("Input: Lengths must be a uint32_t");
            }
            return Input::refString(
                    data.data(),
                    data.nbytes(),
                    static_cast<const uint32_t*>(lengths->data()),
                    lengths->size());
    }
    throw Exception("Input: Unknown type");
}

Output outputFromArrays(
        Type type,
        const nb::ndarray<nb::device::cpu, nb::c_contig>& data,
        const poly::optional<
                nb::ndarray<nb::device::cpu, nb::c_contig, nb::ndim<1>>>&
                lengths)
{
    if (lengths.has_value() && type != Type::String) {
        throw Exception("Output: Passed lengths to non-string type");
    }
    if (data.ndim() != 1 && type != Type::Struct) {
        throw Exception(
                "Output: Non-struct types takes a 1-dimensional byte array");
    }
    if (data.dtype() != nb::dtype<uint8_t>() && type != Type::Numeric) {
        throw Exception("Output: Passed non-bytes data to non-numeric type");
    }
    switch (type) {
        case Type::Serial:
            return Output::wrapSerial(data.data(), data.size());
        case Type::Struct:
            if (data.ndim() != 2) {
                throw Exception(
                        "Output: Struct takes a 2-dimensional byte array");
            }
            return Output::wrapStruct(
                    data.data(), data.shape(0), data.shape(1));
        case Type::Numeric: {
            auto bits = data.dtype().bits;
            if (bits == 8) {
                return Output::wrapNumeric(data.data(), 1, data.size());
            } else if (bits == 16) {
                return Output::wrapNumeric(data.data(), 2, data.size());
            } else if (bits == 32) {
                return Output::wrapNumeric(data.data(), 4, data.size());
            } else if (bits == 64) {
                return Output::wrapNumeric(data.data(), 8, data.size());
            } else {
                throw Exception(
                        "Output: Numeric input takes 8-, 16-, 32-, or 64-bit data");
            }
        }
        case Type::String:
            if (!lengths.has_value()) {
                throw Exception("Output: Lengths not passed to string type");
            }
            if (lengths->dtype() != nb::dtype<uint32_t>()) {
                throw Exception("Output: Lengths must be a uint32_t");
            }
            throw Exception("Output::wrapString not supported yet");
    }
    throw Exception("Output: Unknown type");
}

template <typename T, typename... NDArrayArgs>
class PyBufferBase : public nb::intrusive_base,
                     public Traversable<PyBufferBase<T, NDArrayArgs...>> {
   public:
    PyBufferBase() = default;

    explicit PyBufferBase(
            T* ptr,
            Type type,
            size_t eltWidth,
            size_t numElts,
            size_t contentSize)
            : ptr_(ptr),
              type_(type),
              eltWidth_(eltWidth),
              numElts_(numElts),
              contentSize_(contentSize)
    {
    }

    template <typename... ExtraNDArrayArgs>
    NDArray<NDArrayArgs..., ExtraNDArrayArgs...> asNDArray() const
    {
        using Ret = NDArray<NDArrayArgs..., ExtraNDArrayArgs...>;
        switch (type_) {
            case Type::Serial:
                assert(eltWidth_ == 1);
                assert(contentSize_ == numElts_ * eltWidth_);
                assert(ptr_ != NULL);
                return Ret(
                        NDArray<NDArrayArgs...,
                                ExtraNDArrayArgs...,
                                nb::ndim<1>,
                                uint8_t>(ptr_, { contentSize_ }));
            case Type::Struct:
                assert(contentSize_ == numElts_ * eltWidth_);
                return Ret(
                        NDArray<NDArrayArgs...,
                                ExtraNDArrayArgs...,
                                nb::ndim<2>,
                                uint8_t>(ptr_, { eltWidth_, numElts_ }));
            case Type::Numeric: {
                if (eltWidth_ == 1) {
                    return Ret(
                            NDArray<NDArrayArgs...,
                                    ExtraNDArrayArgs...,
                                    nb::ndim<1>,
                                    uint8_t>(ptr_, { numElts_ }));
                } else if (eltWidth_ == 2) {
                    return Ret(
                            NDArray<NDArrayArgs...,
                                    ExtraNDArrayArgs...,
                                    nb::ndim<1>,
                                    uint16_t>(ptr_, { numElts_ }));
                } else if (eltWidth_ == 4) {
                    return Ret(
                            NDArray<NDArrayArgs...,
                                    ExtraNDArrayArgs...,
                                    nb::ndim<1>,
                                    uint32_t>(ptr_, { numElts_ }));
                } else {
                    assert(eltWidth_ == 8);
                    return Ret(
                            NDArray<NDArrayArgs...,
                                    ExtraNDArrayArgs...,
                                    nb::ndim<1>,
                                    uint64_t>(ptr_, { numElts_ }));
                }
            }
            case Type::String:
                throw Exception(
                        "Logic error: Cannot get ndarray for string type");
            default:
                throw Exception("Invalid type!");
        }
    }

    T* ptr() const
    {
        return ptr_;
    }
    Type type()
    {
        return type_;
    }
    size_t eltWidth() const
    {
        return eltWidth_;
    }
    size_t numElts() const
    {
        return numElts_;
    }
    size_t contentSize() const
    {
        return contentSize_;
    }

    std::vector<nb::handle> references() const
    {
        if (output_) {
            return { nb::find(*output_) };
        } else {
            return {};
        }
    }

   protected:
    nb::ref<PyOutput> output_{};

   private:
    T* ptr_{};
    Type type_{};
    size_t eltWidth_{};
    size_t numElts_{};
    size_t contentSize_{};
};

class PyBuffer : public PyBufferBase<const void, nb::ro> {
    using Base = PyBufferBase<const void, nb::ro>;
    using Base::Base;

   public:
    static nb::ref<PyBuffer> make(const Input& input)
    {
        return new PyBuffer(
                input.ptr(),
                input.type() == Type::String ? Type::Serial : input.type(),
                input.type() == Type::String ? 1 : input.eltWidth(),
                input.type() == Type::String ? input.contentSize()
                                             : input.numElts(),
                input.contentSize());
    }

    static nb::ref<PyBuffer> make(nb::ref<PyOutput> output)
    {
        nb::ref<PyBuffer> buf = new PyBuffer(
                // TODO(terrelln): Separate ptr() into readPtr and writePtr
                ((const PyOutput*)output.get())->ptr(),
                output->type() == Type::String ? Type::Serial : output->type(),
                output->type() == Type::String ? 1 : output->eltWidth(),
                output->type() == Type::String ? output->contentSize()
                                               : output->numElts(),
                output->contentSize());
        buf->output_ = std::move(output);
        return buf;
    }

    static nb::ref<PyBuffer> make(const uint32_t* lengths, size_t numElts)
    {
        return new PyBuffer(
                lengths,
                Type::Numeric,
                sizeof(*lengths),
                numElts,
                numElts * sizeof(*lengths));
    }

    static nb::ref<PyBuffer>
    make(nb::ref<PyOutput> output, const uint32_t* lengths, size_t numElts)
    {
        auto buf     = PyBuffer::make(lengths, numElts);
        buf->output_ = std::move(output);
        return buf;
    }
};

template <typename Buffer>
nb::class_<Buffer> registerBufferBaseClass(nb::module_& m, const char* name)
{
    return nb::class_<Buffer>(
                   m,
                   name,
                   nb::intrusive_ptr<Buffer>(
                           [](Buffer* o, PyObject* po) noexcept {
                               o->set_self_py(po);
                           }),
                   nb::type_slots(Buffer::typeSlots.data()))
            .def_prop_ro("type", &Buffer::type)
            .def_prop_ro("elt_width", &Buffer::eltWidth)
            .def_prop_ro("num_elts", &Buffer::numElts)
            .def_prop_ro("content_size", &Buffer::contentSize)
            .def(
                    "as_nparray",
                    [](Buffer& buffer) {
                        return buffer.template asNDArray<nb::numpy>();
                    },
                    nb::rv_policy::reference_internal)
            .def(
                    "as_pytensor",
                    [](Buffer& buffer) {
                        return buffer.template asNDArray<nb::pytorch>();
                    },
                    nb::rv_policy::reference_internal)
            .def(
                    "as_dltensor",
                    [](Buffer& buffer) {
                        return buffer.template asNDArray<>();
                    },
                    nb::rv_policy::reference_internal);
}

void registerBufferClass(nb::module_& m)
{
    registerBufferBaseClass<PyBuffer>(m, "Buffer")
            .def("as_bytes", [](PyBuffer& buffer) {
                return nb::bytes(buffer.ptr(), buffer.contentSize());
            });
}

void registerInputClass(nb::module_& m)
{
    nb::class_<PyInput>(
            m,
            "Input",
            nb::intrusive_ptr<PyInput>([](PyInput* o, PyObject* po) noexcept {
                o->set_self_py(po);
            }))
            .def(
                    "__init__",
                    [](PyInput* input,
                       Type type,
                       nb::ndarray<nb::ro, nb::device::cpu, nb::c_contig> data,
                       poly::optional<nb::ndarray<
                               nb::ro,
                               nb::device::cpu,
                               nb::c_contig,
                               nb::ndim<1>>> lengths) {
                        new (input) PyInput(inputFromArrays(
                                type, std::move(data), std::move(lengths)));
                    },
                    nb::arg("type"),
                    nb::arg("data"),
                    nb::arg("lengths") = nb::none())
            .def_prop_ro("type", &PyInput::type)
            .def_prop_ro("num_elts", &PyInput::numElts)
            .def_prop_ro("elt_width", &PyInput::eltWidth)
            .def_prop_ro("content_size", &PyInput::contentSize)
            .def("get_int_metadata", &PyInput::getIntMetadata)
            .def("set_int_metadata", &PyInput::setIntMetadata)
            .def_prop_ro(
                    "content",
                    [](const PyInput& input) { return PyBuffer::make(input); },
                    nb::rv_policy::reference_internal)
            .def_prop_ro(
                    "string_lens",
                    [](const PyInput& input) {
                        return PyBuffer::make(
                                input.stringLens(), input.numElts());
                    },
                    nb::rv_policy::reference_internal);
}

class PyMutBuffer : public PyBufferBase<void> {
   private:
    using PyBufferBase<void>::PyBufferBase;

   public:
    static nb::ref<PyMutBuffer> make(nb::ref<PyOutput> output)
    {
        auto buf = new PyMutBuffer(
                output->ptr(),
                output->type() == Type::String ? Type::Serial : output->type(),
                output->type() == Type::String ? 1 : output->eltWidth(),
                output->type() == Type::String ? output->contentCapacity()
                                               : output->eltsCapacity(),
                output->contentCapacity());
        buf->output_ = output;
        return buf;
    }

    static nb::ref<PyMutBuffer>
    make(nb::ref<PyOutput> output, uint32_t* lengths, size_t numElts)
    {
        auto buf = new PyMutBuffer(
                lengths,
                Type::Numeric,
                sizeof(*lengths),
                numElts,
                numElts * sizeof(*lengths));
        buf->output_ = output;
        return buf;
    }
};

void registerMutBufferClass(nb::module_& m)
{
    registerBufferBaseClass<PyMutBuffer>(m, "MutBuffer");
}

void registerOutputClass(nb::module_& m)
{
    nb::class_<PyOutput>(
            m,
            "Output",
            nb::intrusive_ptr<PyOutput>([](PyOutput* o, PyObject* po) noexcept {
                o->set_self_py(po);
            }))
            .def(
                    "__init__",
                    [](PyOutput* output,
                       Type type,
                       nb::ndarray<nb::device::cpu, nb::c_contig> data,
                       poly::optional<nb::ndarray<
                               nb::device::cpu,
                               nb::c_contig,
                               nb::ndim<1>>> lengths) {
                        new (output) PyOutput(outputFromArrays(
                                type, std::move(data), std::move(lengths)));
                    },
                    nb::arg("type"),
                    nb::arg("data"),
                    nb::arg("lengths") = nb::none())
            .def_prop_ro("type", &PyOutput::type)
            .def_prop_ro("num_elts", &PyOutput::numElts)
            .def_prop_ro("elt_width", &PyOutput::eltWidth)
            .def_prop_ro("content_size", &PyOutput::contentSize)
            .def_prop_ro("elts_capacity", &PyOutput::eltsCapacity)
            .def_prop_ro("content_capacity", &PyOutput::contentCapacity)
            .def("reserve_string_lens", &PyOutput::reserveStringLens)
            .def("get_int_metadata", &PyOutput::getIntMetadata)
            .def("set_int_metadata", &PyOutput::setIntMetadata)
            .def("commit", &PyOutput::commit)
            .def_prop_ro(
                    "content",
                    [](nb::ref<PyOutput> output) {
                        return PyBuffer::make(std::move(output));
                    })
            .def_prop_ro(
                    "mut_content",
                    [](nb::ref<PyOutput> output) {
                        return PyMutBuffer::make(std::move(output));
                    })
            .def_prop_ro(
                    "string_lens",
                    [](nb::ref<PyOutput> output) {
                        return PyBuffer::make(
                                output,
                                // TODO(terrelln): Separate stringLens() into
                                // readStringLens and writeStringLens
                                ((const PyOutput*)output.get())->stringLens(),
                                output->numElts());
                    })
            .def_prop_ro("mut_string_lens", [](nb::ref<PyOutput> output) {
                return PyMutBuffer::make(
                        output, output->stringLens(), output->eltsCapacity());
            });
}

void registerMultiInputCodecDescriptionClass(nb::module_& m)
{
    nb::class_<MultiInputCodecDescription>(m, "MultiInputCodecDescription")
            .def(nb::init<>())
            .def(
                    "__init__",
                    [](MultiInputCodecDescription* desc,
                       unsigned id,
                       poly::optional<std::string> name,
                       std::vector<Type> inputTypes,
                       bool lastInputIsVariable,
                       std::vector<Type> singletonOutputTypes,
                       std::vector<Type> variableOutputTypes) {
                        new (desc) MultiInputCodecDescription{
                            .id                  = id,
                            .name                = std::move(name),
                            .inputTypes          = std::move(inputTypes),
                            .lastInputIsVariable = lastInputIsVariable,
                            .singletonOutputTypes =
                                    std::move(singletonOutputTypes),
                            .variableOutputTypes =
                                    std::move(variableOutputTypes)
                        };
                    },
                    nb::kw_only(),
                    nb::arg("id"),
                    nb::arg("name") = poly::nullopt,
                    nb::arg("input_types"),
                    nb::arg("last_input_is_variable") = false,
                    nb::arg("singleton_output_types"),
                    nb::arg("variable_output_types") = std::vector<Type>{})
            .def_rw("id", &MultiInputCodecDescription::id)
            .def_rw("name", &MultiInputCodecDescription::name)
            .def_rw("input_types", &MultiInputCodecDescription::inputTypes)
            .def_rw("last_input_is_variable",
                    &MultiInputCodecDescription::lastInputIsVariable)
            .def_rw("singleton_output_types",
                    &MultiInputCodecDescription::singletonOutputTypes)
            .def_rw("variable_output_types",
                    &MultiInputCodecDescription::variableOutputTypes);
}

void registerEncoderStateClass(nb::module_& m)
{
    nb::class_<PyEncoderState>(
            m,
            "EncoderState",
            nb::intrusive_ptr<PyEncoderState>(
                    [](PyEncoderState* o, PyObject* po) noexcept {
                        o->set_self_py(po);
                    }))
            .def_prop_ro("inputs", &PyEncoderState::inputs)
            .def("create_output", &PyEncoderState::createOutput)
            .def("get_cparam", &PyEncoderState::getCParam)
            .def("get_local_int_param", &PyEncoderState::getLocalIntParam)
            .def("get_local_param", &PyEncoderState::getLocalParam)
            .def("send_codec_header", &PyEncoderState::sendCodecHeader);
}

void registerCustomEncoderClass(nb::module_& m)
{
    nb::class_<PyCustomEncoder, PyCustomEncoderTrampoline>(
            m,
            "CustomEncoder",
            nb::intrusive_ptr<PyCustomEncoder>(
                    [](PyCustomEncoder* o, PyObject* po) noexcept {
                        o->set_self_py(po);
                    }))
            .def(nb::init<>())
            .def("multi_input_description",
                 &PyCustomEncoder::multiInputDescription)
            .def("encode",
                 nb::overload_cast<nb::ref<PyEncoderState>>(
                         &PyCustomEncoder::encode, nb::const_));
}

class PyDecoderState : public nb::intrusive_base {
   public:
    static nb::ref<PyDecoderState> create(DecoderState& state)
    {
        return new PyDecoderState(state);
    }

    std::vector<nb::ref<const PyInput>> singletonInputs() const
    {
        return singletonInputs_;
    }

    std::vector<nb::ref<const PyInput>> variableInputs() const
    {
        return variableInputs_;
    }

    nb::ref<PyOutput>
    createOutput(size_t idx, size_t maxNumElts, size_t eltWidth)
    {
        auto out = state_->createOutput(idx, maxNumElts, eltWidth);
        return new PyOutput(std::move(out));
    }

    nb::bytes getCodecHeader() const
    {
        auto data = state_->getCodecHeader();
        return nb::bytes(data.data(), data.size());
    }

   private:
    explicit PyDecoderState(DecoderState& state)
            : state_(&state),
              singletonInputs_(toPyInputs(state.singletonInputs())),
              variableInputs_(toPyInputs(state.variableInputs()))
    {
    }
    DecoderState* state_;
    std::vector<nb::ref<const PyInput>> singletonInputs_;
    std::vector<nb::ref<const PyInput>> variableInputs_;
};

void registerDecoderStateClass(nb::module_& m)
{
    nb::class_<PyDecoderState>(
            m,
            "DecoderState",
            nb::intrusive_ptr<PyDecoderState>(
                    [](PyDecoderState* o, PyObject* po) noexcept {
                        o->set_self_py(po);
                    }))
            .def_prop_ro("singleton_inputs", &PyDecoderState::singletonInputs)
            .def_prop_ro("variable_inputs", &PyDecoderState::variableInputs)
            .def("create_output", &PyDecoderState::createOutput)
            .def_prop_ro("codec_header", &PyDecoderState::getCodecHeader);
}

class PyCustomDecoder : public CustomDecoder, public nb::intrusive_base {
   public:
    using CustomDecoder::CustomDecoder;

    virtual void decode(nb::ref<PyDecoderState> state) const = 0;

    void decode(DecoderState& state) const override
    {
        auto pyState = PyDecoderState::create(state);
        return decode(std::move(pyState));
    }
};

class PyCustomDecoderTrampoline : public PyCustomDecoder {
   public:
    NB_TRAMPOLINE(PyCustomDecoder, 2);

    MultiInputCodecDescription multiInputDescription() const override
    {
        NB_OVERRIDE_NAME("multi_input_description", multiInputDescription);
    }

    void decode(nb::ref<PyDecoderState> state) const override
    {
        NB_OVERRIDE_PURE(decode, state);
    }
};

void registerCustomDecoderClass(nb::module_& m)
{
    nb::class_<PyCustomDecoder, PyCustomDecoderTrampoline>(
            m,
            "CustomDecoder",
            nb::intrusive_ptr<PyCustomDecoder>(
                    [](PyCustomDecoder* o, PyObject* po) noexcept {
                        o->set_self_py(po);
                    }))
            .def(nb::init<>())
            .def("multi_input_description",
                 &PyCustomDecoder::multiInputDescription)
            .def("decode",
                 nb::overload_cast<nb::ref<PyDecoderState>>(
                         &PyCustomDecoder::decode, nb::const_));
}

static void addParam(
        LocalParams& params,
        int key,
        const std::variant<int, nb::bytes>& value)
{
    if (const int* intValue = std::get_if<int>(&value)) {
        params.addIntParam(key, *intValue);
    } else if (const nb::bytes* bytesValue = std::get_if<nb::bytes>(&value)) {
        params.addCopyParam(key, bytesValue->data(), bytesValue->size());
    }
}

void registerLocalParamsClass(nb::module_& m)
{
    nb::class_<LocalParams>(m, "LocalParams")
            .def(nb::init())
            .def(
                    "__init__",
                    [](LocalParams* p,
                       const std::unordered_map<
                               int,
                               std::variant<int, nb::bytes>>& params) {
                        new (p) LocalParams();
                        for (const auto& [key, value] : params) {
                            addParam(*p, key, value);
                        }
                    },
                    nb::arg("params"))
            .def("add_param", addParam, nb::arg("key"), nb::arg("value"))
            .def("get_params", [](LocalParams& params) {
                std::unordered_map<int, std::variant<int, nb::bytes>> out;
                for (const auto& [key, value] : params.getIntParams()) {
                    out.emplace(key, value);
                }
                for (const auto& [key, ptr, size] : params.getRefParams()) {
                    out.emplace(key, nb::bytes(ptr, size));
                }
                for (const auto& [key, ptr, size] : params.getCopyParams()) {
                    out.emplace(key, nb::bytes(ptr, size));
                }

                return out;
            });
}

void registerEdgeClass(nb::module_& m)
{
    nb::class_<PyEdge>(
            m,
            "Edge",
            nb::intrusive_ptr<PyEdge>([](PyEdge* o, PyObject* po) noexcept {
                o->set_self_py(po);
            }))
            .def_prop_ro("input", &PyEdge::getInput)
            .def("run_node",
                 &PyEdge::runNode,
                 nb::arg("node"),
                 nb::kw_only(),
                 nb::arg("name")   = poly::nullopt,
                 nb::arg("params") = poly::nullopt)
            .def_static(
                    "run_multi_input_node",
                    &PyEdge::runMultiInputNode,
                    nb::arg("inputs"),
                    nb::arg("node"),
                    nb::kw_only(),
                    nb::arg("name")         = poly::nullopt,
                    nb::arg("local_params") = poly::nullopt)
            .def("set_int_metadata",
                 &PyEdge::setIntMetadata,
                 nb::arg("key"),
                 nb::arg("value"))
            .def("set_destination",
                 &PyEdge::setDestination,
                 nb::arg("graph"),
                 nb::kw_only(),
                 nb::arg("name")          = poly::nullopt,
                 nb::arg("custom_graphs") = poly::nullopt,
                 nb::arg("custom_nodes")  = poly::nullopt,
                 nb::arg("local_params")  = poly::nullopt)
            .def_static(
                    "set_multi_input_destination",
                    PyEdge::setMultiInputDestination,
                    nb::arg("inputs"),
                    nb::arg("graph"),
                    nb::kw_only(),
                    nb::arg("name")          = poly::nullopt,
                    nb::arg("custom_graphs") = poly::nullopt,
                    nb::arg("custom_nodes")  = poly::nullopt,
                    nb::arg("local_params")  = poly::nullopt);
}

void registerGraphStateClass(nb::module_& m)
{
    nb::class_<PyGraphState>(
            m,
            "GraphState",
            nb::intrusive_ptr<PyGraphState>(
                    [](PyGraphState* o, PyObject* po) noexcept {
                        o->set_self_py(po);
                    }))
            .def_prop_ro("edges", &PyGraphState::edges)
            .def_prop_ro("custom_graphs", &PyGraphState::customGraphs)
            .def_prop_ro("custom_nodes", &PyGraphState::customNodes)
            .def("get_cparam", &PyGraphState::getCParam, nb::arg("param"))
            .def("get_local_int_param",
                 &PyGraphState::getLocalIntParam,
                 nb::arg("key"))
            .def("get_local_param",
                 &PyGraphState::getLocalParam,
                 nb::arg("key"))
            .def("is_node_supported",
                 &PyGraphState::isNodeSupported,
                 nb::arg("node"));
}

void registerFunctionGraphDescriptionClass(nb::module_& m)
{
    nb::class_<FunctionGraphDescription>(m, "FunctionGraphDescription")
            .def(
                    "__init__",
                    [](FunctionGraphDescription* desc,
                       poly::optional<std::string> name,
                       std::vector<TypeMask> inputTypeMasks,
                       bool lastInputIsVariable,
                       std::vector<GraphID> customGraphs,
                       std::vector<NodeID> customNodes,
                       poly::optional<LocalParams> localParams) {
                        new (desc) FunctionGraphDescription{
                            .name                = std::move(name),
                            .inputTypeMasks      = std::move(inputTypeMasks),
                            .lastInputIsVariable = lastInputIsVariable,
                            .customGraphs        = std::move(customGraphs),
                            .customNodes         = std::move(customNodes),
                            .localParams         = std::move(localParams),
                        };
                    },
                    nb::kw_only(),
                    nb::arg("name") = poly::nullopt,
                    nb::arg("input_type_masks"),
                    nb::arg("last_input_is_variable") = false,
                    nb::arg("custom_graphs")          = std::vector<GraphID>(),
                    nb::arg("custom_nodes")           = std::vector<NodeID>(),
                    nb::arg("local_params")           = poly::nullopt)
            .def_rw("name", &FunctionGraphDescription::name)
            .def_rw("input_type_masks",
                    &FunctionGraphDescription::inputTypeMasks)
            .def_rw("last_input_is_variable",
                    &FunctionGraphDescription::lastInputIsVariable)
            .def_rw("custom_graphs", &FunctionGraphDescription::customGraphs)
            .def_rw("custom_nodes", &FunctionGraphDescription::customNodes)
            .def_rw("local_params", &FunctionGraphDescription::localParams);
}

void registerFunctionGraphClass(nb::module_& m)
{
    nb::class_<PyFunctionGraph, PyFunctionGraphTrampoline>(
            m,
            "FunctionGraph",
            nb::intrusive_ptr<PyFunctionGraph>(
                    [](PyFunctionGraph* o, PyObject* po) noexcept {
                        o->set_self_py(po);
                    }))
            .def(nb::init<>())
            .def("function_graph_description",
                 &PyFunctionGraph::functionGraphDescription)
            .def("graph",
                 nb::overload_cast<nb::ref<PyGraphState>>(
                         &PyFunctionGraph::graph, nb::const_),
                 nb::arg("state"));
}

void registerSelectorStateClass(nb::module_& m)
{
    nb::class_<PySelectorState>(
            m,
            "SelectorState",
            nb::intrusive_ptr<PySelectorState>(
                    [](PySelectorState* o, PyObject* po) noexcept {
                        o->set_self_py(po);
                    }))
            .def_prop_ro("custom_graphs", &PySelectorState::customGraphs)
            .def("get_cparam", &PySelectorState::getCParam, nb::arg("param"))
            .def("get_local_int_param",
                 &PySelectorState::getLocalIntParam,
                 nb::arg("key"))
            .def("get_local_param",
                 &PySelectorState::getLocalParam,
                 nb::arg("key"))
            .def("parameterize_destination",
                 &PySelectorState::parameterizeDestination,
                 nb::kw_only(),
                 nb::arg("name")          = poly::nullopt,
                 nb::arg("custom_graphs") = poly::nullopt,
                 nb::arg("custom_nodes")  = poly::nullopt,
                 nb::arg("local_params")  = poly::nullopt);
}

void registerSelectorDescriptionClass(nb::module_& m)
{
    nb::class_<SelectorDescription>(m, "SelectorDescription")
            .def(
                    "__init__",
                    [](SelectorDescription* desc,
                       poly::optional<std::string> name,
                       TypeMask inputTypeMask,
                       std::vector<GraphID> customGraphs,
                       poly::optional<LocalParams> localParams) {
                        new (desc) SelectorDescription{
                            .name          = std::move(name),
                            .inputTypeMask = inputTypeMask,
                            .customGraphs  = std::move(customGraphs),
                            .localParams   = std::move(localParams),
                        };
                    },
                    nb::kw_only(),
                    nb::arg("name") = poly::nullopt,
                    nb::arg("input_type_mask"),
                    nb::arg("custom_graphs") = std::vector<GraphID>(),
                    nb::arg("local_params")  = poly::nullopt)
            .def_rw("name", &SelectorDescription::name)
            .def_rw("input_type_masks", &SelectorDescription::inputTypeMask)
            .def_rw("custom_graphs", &SelectorDescription::customGraphs)
            .def_rw("local_params", &SelectorDescription::localParams);
}

void registerSelectorClass(nb::module_& m)
{
    nb::class_<PySelector, PySelectorTrampoline>(
            m,
            "Selector",
            nb::intrusive_ptr<PySelector>(
                    [](PySelector* o, PyObject* po) noexcept {
                        o->set_self_py(po);
                    }))
            .def(nb::init<>())
            .def("selector_description", &PySelector::selectorDescription)
            .def("select",
                 nb::overload_cast<nb::ref<PySelectorState>, const PyInput&>(
                         &PySelector::select, nb::const_),
                 nb::arg("state"),
                 nb::arg("input"));
}

void registerNodeParametersClass(nb::module_& m)
{
    nb::class_<NodeParameters>(m, "NodeParameters")
            .def(nb::init())
            .def_rw("name", &NodeParameters::name)
            .def_rw("local_params", &NodeParameters::localParams);
}

void registerGraphParametersClass(nb::module_& m)
{
    nb::class_<GraphParameters>(m, "GraphParameters")
            .def(nb::init())
            .def_rw("name", &GraphParameters::name)
            .def_rw("custom_graphs", &GraphParameters::customGraphs)
            .def_rw("custom_nodes", &GraphParameters::customNodes)
            .def_rw("local_params", &GraphParameters::localParams);
}

void registerCompressorClass(nb::module_& m)
{
    auto compressor =
            nb::class_<PyCompressor>(
                    m,
                    "Compressor",
                    nb::intrusive_ptr<PyCompressor>(
                            [](PyCompressor* o, PyObject* po) noexcept {
                                o->set_self_py(po);
                            }),
                    nb::type_slots(PyCompressor::typeSlots.data()))
                    .def(nb::init())
                    .def("set_parameter", &PyCompressor::setParameter)
                    .def("get_parameter", &PyCompressor::getParameter)
                    .def("build_static_graph",
                         &PyCompressor::buildStaticGraph,
                         nb::arg("head_node"),
                         nb::arg("successor_graphs"),
                         nb::kw_only(),
                         nb::arg("name")         = poly::nullopt,
                         nb::arg("local_params") = poly::nullopt)
                    .def("parameterize_node",
                         &PyCompressor::parameterizeNode,
                         nb::arg("node"),
                         nb::kw_only(),
                         nb::arg("name")         = poly::nullopt,
                         nb::arg("local_params") = poly::nullopt)
                    .def("parameterize_graph",
                         &PyCompressor::parameterizeGraph,
                         nb::arg("graph"),
                         nb::kw_only(),
                         nb::arg("name")          = poly::nullopt,
                         nb::arg("custom_graphs") = poly::nullopt,
                         nb::arg("custom_nodes")  = poly::nullopt,
                         nb::arg("local_params")  = poly::nullopt)
                    .def("register_custom_encoder",
                         &PyCompressor::registerCustomEncoder)
                    .def("register_function_graph",
                         &PyCompressor::registerFunctionGraph)
                    .def("register_selector_graph",
                         &PyCompressor::registerSelectorGraph)
                    .def("get_node",
                         nb::overload_cast<const std::string&>(
                                 &PyCompressor::getNode, nb::const_),
                         nb::arg("name"))
                    .def("get_graph",
                         nb::overload_cast<const std::string&>(
                                 &PyCompressor::getGraph, nb::const_),
                         nb::arg("name"))
                    .def("select_starting_graph",
                         &PyCompressor::selectStartingGraph,
                         nb::arg("graph"))
                    .def("serialize", &PyCompressor::serialize)
                    .def("serialize_to_json", &PyCompressor::serializeToJson)
                    .def("deserialize",
                         &PyCompressor::deserialize,
                         nb::arg("serialized"))
                    .def("get_unmet_dependencies",
                         &PyCompressor::getUnmetDependencies,
                         nb::arg("serialized"));
    nb::class_<PyCompressor::UnmetDependencies>(compressor, "UnmetDependencies")
            .def_ro("graph_names", &PyCompressor::UnmetDependencies::graphNames)
            .def_ro("node_names", &PyCompressor::UnmetDependencies::nodeNames);
}

class PyCCtx : public CCtx,
               public Traversable<PyCCtx>,
               public nb::intrusive_base {
   public:
    using CCtx::CCtx;

    void refCompressor(nb::ref<PyCompressor> compressor)
    {
        this->CCtx::refCompressor(*compressor);
        compressor_ = std::move(compressor);
    }

    void selectStartingGraph(
            nb::ref<PyCompressor> compressor,
            GraphID graph,
            poly::optional<std::string> name,
            poly::optional<std::vector<GraphID>> customGraphs,
            poly::optional<std::vector<NodeID>> customNodes,
            poly::optional<LocalParams> localParams)
    {
        this->CCtx::selectStartingGraph(
                *compressor,
                graph,
                GraphParameters{ .name         = std::move(name),
                                 .customGraphs = std::move(customGraphs),
                                 .customNodes  = std::move(customNodes),
                                 .localParams  = std::move(localParams) });
        compressor_ = std::move(compressor);
    }

    nb::bytes compress(std::vector<nb::ref<PyInput>> inputs)
    {
        std::vector<InputRef> refs;
        refs.reserve(inputs.size());
        for (auto& input : inputs) {
            refs.emplace_back(input->get());
        }
        auto compressed = this->CCtx::compress(
                poly::span<const Input>(
                        static_cast<Input*>(refs.data()), refs.size()));
        return nb::bytes(compressed.data(), compressed.size());
    }

    std::vector<nb::handle> references() const
    {
        if (compressor_) {
            return { nb::find(compressor_) };
        } else {
            return {};
        }
    }

   private:
    nb::ref<PyCompressor> compressor_;
};

void registerCCtxClass(nb::module_& m)
{
    nb::class_<PyCCtx>(
            m,
            "CCtx",
            nb::intrusive_ptr<PyCCtx>([](PyCCtx* o, PyObject* po) noexcept {
                o->set_self_py(po);
            }),
            nb::type_slots(PyCCtx::typeSlots.data()))
            .def(nb::init())
            .def("ref_compressor", &PyCCtx::refCompressor)
            .def("set_parameter", &PyCCtx::setParameter)
            .def("get_parameter", &PyCCtx::getParameter)
            .def("reset_parameters", &PyCCtx::resetParameters)
            .def("select_starting_graph",
                 &PyCCtx::selectStartingGraph,
                 nb::arg("compressor"),
                 nb::arg("graph"),
                 nb::kw_only(),
                 nb::arg("name")          = poly::nullopt,
                 nb::arg("custom_graphs") = poly::nullopt,
                 nb::arg("custom_nodes")  = poly::nullopt,
                 nb::arg("local_params")  = poly::nullopt)
            .def("compress", &PyCCtx::compress);
}

class PyFrameInfo : public FrameInfo, public nb::intrusive_base {
   public:
    explicit PyFrameInfo(const nb::bytes& data)
            : FrameInfo({ static_cast<const char*>(data.data()), data.size() })
    {
    }
};

void registerFrameInfoClass(nb::module_& m)
{
    nb::class_<PyFrameInfo>(
            m,
            "FrameInfo",
            nb::intrusive_ptr<PyFrameInfo>(
                    [](PyFrameInfo* o, PyObject* po) noexcept {
                        o->set_self_py(po);
                    }))
            .def(nb::init<const nb::bytes&>())
            .def_prop_ro("num_outputs", &PyFrameInfo::numOutputs)
            .def("output_type", &PyFrameInfo::outputType, nb::arg("index"))
            .def("output_content_size",
                 &PyFrameInfo::outputContentSize,
                 nb::arg("index"));
}

class PyDCtx : public DCtx,
               public Traversable<PyDCtx>,
               public nb::intrusive_base {
   public:
    using DCtx::DCtx;
    using PyBufferInput = nb::ndarray<const uint8_t, nb::ndim<1>, nb::c_contig, nb::memview, nb::ro>;

    std::vector<nb::ref<PyOutput>> decompress(PyBufferInput buffer)
    {
        const char* input_data = reinterpret_cast<const char*>(buffer.data());
        size_t input_size = buffer.size();

        auto out = this->DCtx::decompress({ input_data, input_size });

        std::vector<nb::ref<PyOutput>> pyOut;
        pyOut.reserve(out.size());
        for (auto& o : out) {
            pyOut.push_back(new PyOutput(std::move(o)));
        }
        return pyOut;
    }

    void registerCustomDecoder(nb::ref<PyCustomDecoder> decoder)
    {
        reference(*decoder);
        auto raw           = decoder.get();
        auto sharedDecoder = std::shared_ptr<CustomDecoder>(
                raw, [decoder = std::move(decoder)](CustomDecoder* ptr) {});
        this->DCtx::registerCustomDecoder(std::move(sharedDecoder));
    }

    std::vector<nb::handle> references() const
    {
        return references_;
    }

   private:
    template <typename T>
    void reference(const T& val)
    {
        nb::handle obj = nb::find(val);
        if (obj.ptr() != nullptr) {
            references_.push_back(std::move(obj));
        }
    }

    std::vector<nb::handle> references_;
};

void registerDCtxClass(nb::module_& m)
{
    nb::class_<PyDCtx>(
            m,
            "DCtx",
            nb::intrusive_ptr<PyDCtx>([](PyDCtx* o, PyObject* po) noexcept {
                o->set_self_py(po);
            }),
            nb::type_slots(PyDCtx::typeSlots.data()))
            .def(nb::init())
            .def("set_parameter", &PyDCtx::setParameter)
            .def("get_parameter", &PyDCtx::getParameter)
            .def("reset_parameters", &PyDCtx::resetParameters)
            .def("decompress", &PyDCtx::decompress, nb::arg("compressed").noconvert())
            .def("register_custom_decoder", &PyDCtx::registerCustomDecoder);
}

void registerSysModule(nb::module_& m)
{
    registerVersionInfo(m);
    registerNodeID(m);
    registerGraphID(m);
    registerType(m);
    registerTypeMask(m);
    registerCParam(m);
    registerDParam(m);
    registerBufferClass(m);
    registerInputClass(m);
    registerMutBufferClass(m);
    registerOutputClass(m);
    registerNodeParametersClass(m);
    registerGraphParametersClass(m);
    registerCompressorClass(m);
    registerCCtxClass(m);
    registerDCtxClass(m);
    registerFrameInfoClass(m);
    registerMultiInputCodecDescriptionClass(m);
    registerEncoderStateClass(m);
    registerCustomEncoderClass(m);
    registerDecoderStateClass(m);
    registerCustomDecoderClass(m);
    registerEdgeClass(m);
    registerGraphStateClass(m);
    registerFunctionGraphDescriptionClass(m);
    registerFunctionGraphClass(m);
    registerSelectorStateClass(m);
    registerSelectorDescriptionClass(m);
    registerSelectorClass(m);
    registerLocalParamsClass(m);
    registerNodesModule(m);
    registerGraphsModule(m);
}
} // namespace
} // namespace py
} // namespace openzl

NB_MODULE(ext, m)
{
    nanobind::intrusive_init(
            [](PyObject* o) noexcept {
                nanobind::gil_scoped_acquire guard;
                Py_INCREF(o);
            },
            [](PyObject* o) noexcept {
                nanobind::gil_scoped_acquire guard;
                Py_DECREF(o);
            });
    openzl::py::registerSysModule(m);
}
