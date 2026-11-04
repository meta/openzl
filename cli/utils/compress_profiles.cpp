// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cli/utils/compress_profiles.h"
#include "cli/utils/profile_graphs.h"
#include "cli/utils/util.h"

#include <string.h>

#include <limits>

#include "openzl/codecs/zl_conversion.h"
#include "openzl/codecs/zl_generic.h"
#include "openzl/codecs/zl_mlselector.h"
#include "openzl/codecs/zl_sddl2.h"
#include "openzl/common/assertion.h"
#include "openzl/cpp/Exception.hpp"
#include "openzl/openzl.hpp"
#include "openzl/zl_compressor.h"
#include "openzl/zl_reflection.h"

#include "custom_parsers/csv/csv_profile.h"
#include "custom_parsers/dependency_registration.h"
#include "custom_parsers/gguf/gguf_parser.h"
#include "custom_parsers/numpy/numpy_parser.h"
#include "custom_parsers/onnx/onnx_parser.h"
#include "custom_parsers/parquet/parquet_graph.h"
#include "custom_parsers/pytorch_model_parser.h"
#include "custom_parsers/safetensors/safetensors_parser.h"
#include "custom_parsers/sddl/sddl_profile.h"
#include "custom_parsers/shared_components/clustering.h"

#include "tools/io/InputFile.h"
#include "tools/io/InputSetFileOrDir.h"
#include "tools/sddl/compiler/Compiler.h"

#include "tools/sddl2/assembler/Assembler.h"
#include "tools/sddl2/compiler/Compiler.h"

namespace openzl::cli {

void ProfileArgs::addArgs(arg::ArgParser& parser)
{
    parser.addGlobalFlag(kProfile, 'p', true, "Select the given profile.");
    parser.addGlobalFlag(
            kProfileArg,
            0,
            true,
            "Pass the given value as an argument to constructing the profile.");
    parser.addGlobalFlag(
            kChunkSize,
            0,
            true,
            "The chunk size for the input to be separated into (e.g. 20M, 500K, 1G). Supports suffixes: K/KB (10^3), M/MB (10^6), G/GB (10^9), T/TB (10^12), and binary KiB (2^10), MiB (2^20), GiB (2^30), TiB (2^40). Plain numbers are treated as bytes. When omitted, profiles use their built-in default chunk size if they segment input.");
}

ProfileArgs::ProfileArgs(const arg::ParsedArgs& parsed)
{
    auto chunkSize = parsed.globalFlag(kChunkSize);
    if (chunkSize.has_value()) {
        chunkSize_ = util::checkedstoul(chunkSize.value());
    } else {
        chunkSize_ = poly::nullopt;
    }
    auto profileArg = parsed.globalFlag(kProfileArg);
    if (profileArg) {
        argmap_.emplace("TBD", profileArg.value());
    }
    auto profile = parsed.globalFlag(kProfile);
    if (profile) {
        name_ = std::move(profile);
    }
}

namespace {
std::array<ZL_GraphID, 6> saoTransformerFieldGraphs(Compressor& compressor)
{
    ZL_GraphID const numeric =
            nodes::ConvertStructToNumLE()(compressor, ZL_GRAPH_NUMERIC);
    /* Each field needs a distinct ACE wrapper so it can be trained
     * independently, even though their initial graphs are identical. */
    return {
        graphs::ACE(numeric)(compressor), graphs::ACE(numeric)(compressor),
        graphs::ACE(numeric)(compressor), graphs::ACE(numeric)(compressor),
        graphs::ACE(numeric)(compressor), graphs::ACE(numeric)(compressor)
    };
}

std::array<ZL_GraphID, 6> saoTunedFieldGraphs(Compressor& compressor)
{
    ZL_GraphID const sra0 = graphs::ACE(
            nodes::ConvertStructToNumLE()(
                    compressor,
                    nodes::DeltaInt()(
                            compressor, graphs::FieldLz()(compressor))))(
            compressor);
    ZL_GraphID const sdec0 = graphs::ACE(
            nodes::TransposeSplit()(compressor, graphs::Zstd()(compressor)))(
            compressor);
    ZL_GraphID const tokenCompress = nodes::TokenizeStruct()(
            compressor,
            graphs::FieldLz()(compressor),
            graphs::FieldLz()(compressor));
    ZL_GraphID const numHuffman = nodes::ConvertStructToNumLE()(
            compressor,
            nodes::TokenizeNumeric(/* sort */ false)(
                    compressor,
                    graphs::Huffman()(compressor),
                    graphs::Huffman()(compressor)));

    ZL_GraphID const is   = graphs::ACE(numHuffman)(compressor);
    ZL_GraphID const mag  = graphs::ACE(numHuffman)(compressor);
    ZL_GraphID const xrpm = graphs::ACE(tokenCompress)(compressor);
    ZL_GraphID const xdpm = graphs::ACE(tokenCompress)(compressor);
    return { sra0, sdec0, is, mag, xrpm, xdpm };
}

ZL_GraphID saoProfile(Compressor& compressor, const ProfileArgs& args)
{
    /* This currently matches the Standard Numeric Selector rollout threshold,
     * ensuring the high-level SAO branch reaches Transformer. Reevaluate the
     * relationship if either rollout policy changes. */
    constexpr int kTransformerMinCompressionLevel = 7;
    /* Preserve SAO's historical level-1 default when no level was requested;
     * an explicit CLI level remains authoritative. */
    int const compressionLevel = args.requestedCompressionLevel().value_or(1);
    compressor.setParameter(CParam::CompressionLevel, compressionLevel);
    /* The SAO format consists of a header,
     * which is 28 bytes for the dirSilesia/sao sample specifically,
     * followed by an array of structures, each one describing a star.
     *
     * For the record, here is the Header format (it's currently ignored):
     *
     * Integer*4 STAR0=0   Subtract from star number to get sequence number
     * Integer*4 STAR1=1   First star number in file
     * Integer*4 STARN=258996  Number of stars in file (pos 8)
     * Integer*4 STNUM=1   0 if no star i.d. numbers are present
     *                     1 if star i.d. numbers are in catalog file
     *                     2 if star i.d. numbers are  in file
     * Logical*4 MPROP=t   True if proper motion is included
     *                     False if no proper motion is included
     * Integer*4 NMAG=1    Number of magnitudes present
     * Integer*4 NBENT=32  Number of bytes per star entry
     * Total : 28 bytes
     */
    size_t const headerSize = 28;

    /* Star record : 28 bytes for the dirSilesia/sao sample specifically
     * Real*4 XNO       Catalog number of star (not present, since stnum==0)
     * Real*8 SRA0      B1950 Right Ascension (radians)
     * Real*8 SDEC0     B1950 Declination (radians)
     * Character*2 IS   Spectral type (2 characters)
     * Integer*2 MAG    V Magnitude * 100
     * Real*4 XRPM      R.A. proper motion (radians per year)
     * Real*4 XDPM      Dec. proper motion (radians per year)
     */
    std::array<size_t, 6> const fieldSizes = { 8, 8, 2, 2, 4, 4 };
    std::array<ZL_GraphID, 6> const fieldGraphs =
            compressionLevel >= kTransformerMinCompressionLevel
            ? saoTransformerFieldGraphs(compressor)
            : saoTunedFieldGraphs(compressor);

    ZL_GraphID splitStructure = ZL_Compressor_registerSplitByStructGraph(
            compressor.get(),
            fieldSizes.data(),
            fieldGraphs.data(),
            fieldSizes.size());

    const std::array<size_t, 2> splitSizes      = { headerSize, 0 };
    const std::array<ZL_GraphID, 2> splitGraphs = { ZL_GRAPH_STORE,
                                                    splitStructure };

    return ZL_Compressor_registerSplitGraph(
            compressor.get(),
            ZL_Type_serial,
            splitSizes.data(),
            splitGraphs.data(),
            splitSizes.size());
}

struct IntProfileData {
    size_t eltByteWidth;
    ZL_NodeID conversionNode;
};

enum class IntSignedness { SIGNED, UNSIGNED };
enum class IntEndianness { LITTLE, BIG };

static std::string
makeProfileName(const std::string& signage, size_t bitWidth, bool isBigEndian)
{
    if (bitWidth == 8) {
        return signage + "8";
    }
    return (isBigEndian ? "be-" : "le-") + signage + std::to_string(bitWidth);
}

static ZL_GraphID
buildIntProfile(ZL_Compressor* comp, void* opaque, const ProfileArgs& args)
{
    if (!opaque) {
        return ZL_GRAPH_ILLEGAL;
    }
    const auto& d = *static_cast<const IntProfileData*>(opaque);
    const size_t chunkSize =
            args.chunkSize().value_or(ZL_DEFAULT_SEGMENTER_CHUNK_BYTE_SIZE);
    return profiles::buildIntGraph(
            comp, d.eltByteWidth, d.conversionNode, chunkSize);
}

static void addIntProfile(
        std::map<std::string, std::shared_ptr<CompressProfile>>& mp,
        IntSignedness signedness,
        size_t bitWidth,
        IntEndianness endianness = IntEndianness::LITTLE)
{
    // Signedness affects only profile metadata; signed and unsigned profiles
    // intentionally use the same bit-preserving numeric graph.
    bool const isSigned    = signedness == IntSignedness::SIGNED;
    bool const isBigEndian = endianness == IntEndianness::BIG;
    // Byte order is meaningless for 8-bit values; register them once without
    // an endian prefix rather than creating duplicate little- and big-endian
    // aliases.
    ZL_ASSERT_OR(bitWidth != 8, !isBigEndian);
    const std::string signage = isSigned ? "i" : "u";
    const std::string name    = makeProfileName(signage, bitWidth, isBigEndian);
    const ZL_NodeID conversionNode = isBigEndian
            ? ZL_Node_convertSerialToNumBE(bitWidth)
            : ZL_Node_convertSerialToNumLE(bitWidth);

    auto data = std::make_shared<IntProfileData>(
            IntProfileData{ bitWidth / 8, conversionNode });

    mp[name] = std::make_shared<CompressProfile>(
            name,
            profiles::getProfileDescription(name),
            buildIntProfile,
            std::move(data),
            true);
}

/// The numpy and pickle profiles build the same graph.
static void addNumPyProfile(
        std::map<std::string, std::shared_ptr<CompressProfile>>& mp,
        const std::string& name,
        const std::string& description)
{
    mp[name] = std::make_shared<CompressProfile>(
            name,
            description,
            [name](ZL_Compressor* comp, void*, const ProfileArgs& args) {
                const auto graph = ZL_NumPy_registerGraph(
                        comp,
                        args.chunkSize().value_or(ZL_NUMPY_DEFAULT_CHUNK_SIZE));
                if (ZL_RES_isError(graph)) {
                    throw InvalidArgsException(
                            "Failed to build the " + name + " profile: "
                            + ZL_Compressor_getErrorContextString_fromError(
                                    comp, ZL_RES_error(graph)));
                }
                return ZL_RES_value(graph);
            },
            nullptr,
            true);
}
} // namespace

static ZL_RESULT_OF(ZL_GraphID) extractFolderOfCompressors(
        Compressor& compressor,
        const std::string& folder)
{
    ZL_RESULT_DECLARE_SCOPE(ZL_GraphID, compressor.get());

    auto inputSet = tools::io::InputSetFileOrDir(folder, true);

    std::vector<ZL_GraphID> successors;
    for (const auto& input : inputSet) {
        auto contents = input->contents();

        const auto deps = compressor.getUnmetDependencies(contents);

        for (const auto& graphName : deps.graphNames) {
            if (graphName == "Parquet Parser" || graphName == "CSV Parser") {
                throw std::runtime_error(
                        "CSV and Parquet parsers are not supported in ML Selector profiles. "
                        "CSV and Parquet parsers require clustering which ML Selector does not provide.");
            }
        }

        compressor.deserialize(contents);

        ZL_GraphID startingGraphId;
        ZL_Compressor_getStartingGraphID(compressor.get(), &startingGraphId);
        successors.push_back(startingGraphId);
    }

    // ML selectors require at least 2 successors to choose between
    if (successors.size() < 2) {
        throw Exception(
                "ML selector requires at least 2 successor compressors, but "
                "only "
                + std::to_string(successors.size()) + " were provided in '"
                + folder + "'");
    }

    ZL_TRY_LET(
            ZL_GraphID,
            mlSelectorGraphId,
            ZL_Compressor_buildUntrainedMLSelector(
                    compressor.get(), successors.data(), successors.size()));

    // Wrap with serial-to-numeric conversion so the graph accepts serial input
    ZL_GraphID staticGraph = ZL_Compressor_registerStaticGraph_fromNode1o(
            compressor.get(),
            ZL_NODE_CONVERT_SERIAL_TO_NUM_LE64,
            mlSelectorGraphId);

    // Parameterize so ml selector graph can be updated during training
    ZL_GraphParameters const wrapperDesc = {};

    return ZL_Compressor_parameterizeGraph(
            compressor.get(), staticGraph, &wrapperDesc);
}

/**
 * @brief Registers static successor graphs for the ML selector.
 * NOTE: This is a temporary placeholder
 * @param compressor The compressor to register the graph with
 */
static ZL_RESULT_OF(ZL_GraphID)
        numeric64BitMLSelectorProfile(ZL_Compressor* compressor)
{
    ZL_RESULT_DECLARE_SCOPE(ZL_GraphID, compressor);

    ZL_GraphID fieldlz    = ZL_Compressor_registerFieldLZGraph(compressor);
    ZL_GraphID range_pack = ZL_Compressor_registerStaticGraph_fromNode(
            compressor, ZL_NODE_RANGE_PACK, &fieldlz, 1);
    ZL_GraphID zstd            = ZL_GRAPH_ZSTD;
    ZL_GraphID range_pack_zstd = ZL_Compressor_registerStaticGraph_fromNode(
            compressor, ZL_NODE_RANGE_PACK, &zstd, 1);
    ZL_GraphID delta_fieldlz = ZL_Compressor_registerStaticGraph_fromNode(
            compressor, ZL_NODE_DELTA_INT, &fieldlz, 1);
    ZL_GraphID tokenize_delta_fieldlz = ZL_Compressor_registerTokenizeGraph(
            compressor,
            ZL_Type_numeric,
            /* sort */ true,
            delta_fieldlz,
            fieldlz);

    ZL_GraphID successors[6] = { fieldlz,
                                 range_pack,
                                 range_pack_zstd,
                                 delta_fieldlz,
                                 tokenize_delta_fieldlz,
                                 zstd };

    ZL_TRY_LET(
            ZL_GraphID,
            mlSelectorGraphId,
            ZL_Compressor_buildUntrainedMLSelector(compressor, successors, 6));

    // Wrap with serial-to-numeric conversion so the graph accepts serial input
    ZL_GraphID staticGraph = ZL_Compressor_registerStaticGraph_fromNode1o(
            compressor, ZL_NODE_CONVERT_SERIAL_TO_NUM_LE64, mlSelectorGraphId);

    // Parameterize so ml selector graph can be updated during training
    ZL_GraphParameters const wrapperDesc = {};

    return ZL_Compressor_parameterizeGraph(
            compressor, staticGraph, &wrapperDesc);
}

const std::map<std::string, std::shared_ptr<CompressProfile>>&
compressProfiles()
{
    using ProfileMap = std::map<std::string, std::shared_ptr<CompressProfile>>;
    static const ProfileMap staticProfiles = []() {
        std::map<std::string, std::shared_ptr<CompressProfile>> mp;

        std::string kSerialName = "serial";
        mp[kSerialName]         = std::make_shared<CompressProfile>(
                kSerialName,
                profiles::getProfileDescription(kSerialName),
                [](ZL_Compressor* compressor, void*, const ProfileArgs& args) {
                    size_t chunkSize = args.chunkSize().value_or(
                            ZL_DEFAULT_SEGMENTER_CHUNK_BYTE_SIZE);
                    return profiles::buildSerialGraph(compressor, chunkSize);
                },
                nullptr,
                true);

        const std::string kLz = "lz";
        mp[kLz]               = std::make_shared<CompressProfile>(
                kLz,
                profiles::getProfileDescription(kLz),
                [](ZL_Compressor* compressor, void*, const ProfileArgs& args) {
                    size_t chunkSize = args.chunkSize().value_or(
                            ZL_DEFAULT_SEGMENTER_CHUNK_BYTE_SIZE);
                    return profiles::buildLzGraph(compressor, chunkSize);
                },
                nullptr,
                true);

        std::string kZstd = "zstd";
        mp[kZstd]         = std::make_shared<CompressProfile>(
                kZstd,
                profiles::getProfileDescription(kZstd),
                [](ZL_Compressor* compressor, void*, const ProfileArgs&) {
                    return profiles::buildZstdGraph(compressor);
                },
                nullptr,
                false);

        std::string kPytorchName = "pytorch";
        mp[kPytorchName]         = std::make_shared<CompressProfile>(
                kPytorchName,
                profiles::getProfileDescription(kPytorchName),
                [](ZL_Compressor* comp, void*, const ProfileArgs&) {
                    return profiles::buildPytorchGraph(comp);
                });

        std::string kCsvName = "csv";
        mp[kCsvName]         = std::make_shared<CompressProfile>(
                kCsvName,
                profiles::getProfileDescription(kCsvName),
                [](ZL_Compressor* comp, void*, const ProfileArgs& args) {
                    char sep             = ',';
                    const auto chunkSize = args.chunkSize().value_or(
                            custom_parsers::kDefaultChunkSize);
                    auto argmap = args.map();
                    auto it     = argmap.find("TBD");
                    if (it != argmap.end()) {
                        auto str = it->second;
                        if (str.size() != 1) {
                            throw InvalidArgsException(
                                    "The CSV profile separator must be a single character. Pass it with --profile-arg <char>.");
                        }
                        sep = str[0];
                    }
                    return profiles::buildCsvGraph(comp, chunkSize, sep);
                },
                nullptr,
                true);

        std::string kSafetensorsName = "safetensors";
        mp[kSafetensorsName]         = std::make_shared<CompressProfile>(
                kSafetensorsName,
                "Safetensors model checkpoint (e.g. Hugging Face model.safetensors).",
                [](ZL_Compressor* comp, void*, const ProfileArgs& args) {
                    const auto graph = ZL_Safetensors_registerGraph(
                            comp,
                            args.chunkSize().value_or(
                                    ZL_SAFETENSORS_DEFAULT_CHUNK_SIZE));
                    if (ZL_RES_isError(graph)) {
                        throw InvalidArgsException(
                                std::string(
                                        "Failed to build the safetensors profile: ")
                                + ZL_Compressor_getErrorContextString_fromError(
                                        comp, ZL_RES_error(graph)));
                    }
                    return ZL_RES_value(graph);
                },
                nullptr,
                true);

        std::string kGGUFName = "gguf";
        mp[kGGUFName]         = std::make_shared<CompressProfile>(
                kGGUFName,
                "GGUF model file (llama.cpp). Quantized weights gain a few percent over zstd, but decompress slower.",
                [](ZL_Compressor* comp, void*, const ProfileArgs& args) {
                    const auto graph = ZL_GGUF_registerGraph(
                            comp,
                            args.chunkSize().value_or(
                                    ZL_GGUF_DEFAULT_CHUNK_SIZE));
                    if (ZL_RES_isError(graph)) {
                        throw InvalidArgsException(
                                std::string(
                                        "Failed to build the gguf profile: ")
                                + ZL_Compressor_getErrorContextString_fromError(
                                        comp, ZL_RES_error(graph)));
                    }
                    return ZL_RES_value(graph);
                },
                nullptr,
                true);

        std::string kONNXName = "onnx";
        mp[kONNXName]         = std::make_shared<CompressProfile>(
                kONNXName,
                "ONNX model file. Models of 2 GB or more keep their weights in external files, which this profile does not parse.",
                [](ZL_Compressor* comp, void*, const ProfileArgs& args) {
                    const auto graph = ZL_ONNX_registerGraph(
                            comp,
                            args.chunkSize().value_or(
                                    ZL_ONNX_DEFAULT_CHUNK_SIZE));
                    if (ZL_RES_isError(graph)) {
                        throw InvalidArgsException(
                                std::string(
                                        "Failed to build the onnx profile: ")
                                + ZL_Compressor_getErrorContextString_fromError(
                                        comp, ZL_RES_error(graph)));
                    }
                    return ZL_RES_value(graph);
                },
                nullptr,
                true);

        addNumPyProfile(
                mp,
                "numpy",
                "NumPy array (.npy), archive of arrays (.npz), or pickle holding NumPy arrays. Arrays deflated by np.savez_compressed are not inflated, so they gain nothing.");
        addNumPyProfile(
                mp,
                "pickle",
                "Python pickle (protocols 2 to 5). The NumPy arrays it holds, such as those of pandas DataFrames, are compressed as numbers.");

        addIntProfile(mp, IntSignedness::SIGNED, 8);
        addIntProfile(mp, IntSignedness::UNSIGNED, 8);
        addIntProfile(mp, IntSignedness::SIGNED, 16);
        addIntProfile(mp, IntSignedness::UNSIGNED, 16);
        addIntProfile(mp, IntSignedness::SIGNED, 32);
        addIntProfile(mp, IntSignedness::UNSIGNED, 32);
        addIntProfile(mp, IntSignedness::SIGNED, 64);
        addIntProfile(mp, IntSignedness::UNSIGNED, 64);
        addIntProfile(mp, IntSignedness::SIGNED, 16, IntEndianness::BIG);
        addIntProfile(mp, IntSignedness::UNSIGNED, 16, IntEndianness::BIG);
        addIntProfile(mp, IntSignedness::SIGNED, 32, IntEndianness::BIG);
        addIntProfile(mp, IntSignedness::UNSIGNED, 32, IntEndianness::BIG);
        addIntProfile(mp, IntSignedness::SIGNED, 64, IntEndianness::BIG);
        addIntProfile(mp, IntSignedness::UNSIGNED, 64, IntEndianness::BIG);

        std::string kParquetName = "parquet";
        mp[kParquetName]         = std::make_shared<CompressProfile>(
                kParquetName,
                profiles::getProfileDescription(kParquetName),
                [](ZL_Compressor* comp, void*, const ProfileArgs& args) {
                    const size_t chunkSize = args.chunkSize().value_or(
                            custom_parsers::kDefaultChunkSize);
                    if (chunkSize > static_cast<size_t>(
                                std::numeric_limits<int>::max())) {
                        throw InvalidArgsException(
                                "--chunk-size for the parquet profile must be at most "
                                + std::to_string(
                                        std::numeric_limits<int>::max())
                                + " bytes.");
                    }
                    return profiles::buildParquetGraph(comp, chunkSize);
                },
                nullptr,
                true);

        std::string kSDDLName = "sddl";
        mp[kSDDLName]         = std::make_shared<CompressProfile>(
                kSDDLName,
                profiles::getProfileDescription(kSDDLName),
                [](ZL_Compressor* comp, void*, const ProfileArgs& args) {
                    auto argmap = args.map();
                    auto it     = argmap.find("TBD");
                    if (it == argmap.end()) {
                        throw InvalidArgsException(
                                "The Simple Data Description Language profile requires a data description. Pass a path to the description file with --profile-arg.");
                    }
                    auto progInput = tools::io::InputFile(it->second);
                    auto compiled  = sddl::Compiler{}.compile(
                            progInput.contents(), progInput.name());
                    return unwrap(
                            ZL_SDDL_setupProfile(
                                    comp, compiled.data(), compiled.size()),
                            "Failed to set up SDDL profile",
                            comp);
                });

        std::string kSDDL2Name = "sddl2";
        mp[kSDDL2Name]         = std::make_shared<CompressProfile>(
                kSDDL2Name,
                profiles::getProfileDescription(kSDDL2Name),
                [](ZL_Compressor* comp, void*, const ProfileArgs& args) {
                    auto argmap = args.map();
                    auto it     = argmap.find("TBD");
                    if (it == argmap.end()) {
                        throw InvalidArgsException(
                                "The Simple Data Description Language v2 profile requires a data description file. Pass a path to the description file with --profile-arg.");
                    }
                    auto description = tools::io::InputFile(it->second);
                    // Map the CLI log level onto the compiler's scale: INFO
                    // (the default) keeps the compiler quiet, and each -v above
                    // INFO raises compiler verbosity by one.
                    auto compiled =
                            sddl2::Compiler{
                                sddl2::Compiler::Options{}.with_verbosity(
                                        args.verbosityLevel() - 3)
                            }
                                    .compile(
                                            description.contents(),
                                            description.name());
                    auto bytecode   = sddl2::Assembler{}.assemble(compiled);
                    auto clustering = ZS2_createGraph_genericClustering(comp);
                    auto graph      = ZL_Compressor_registerSDDL2Graph_advanced(
                            comp,
                            bytecode.data(),
                            bytecode.size(),
                            clustering,
                            args.chunkSize().value_or(0));
                    if (!ZL_GraphID_isValid(graph)) {
                        throw ExceptionBuilder("Failed to set up SDDL2 profile")
                                .withErrorCode(ZL_ErrorCode_graph_invalid)
                                .addErrorContext(comp)
                                .build();
                    }
                    return graph;
                },
                nullptr,
                true);

        std::string kSAOName = "sao";
        mp[kSAOName]         = std::make_shared<CompressProfile>(
                kSAOName,
                profiles::getProfileDescription(kSAOName),
                [](ZL_Compressor* comp, void*, const ProfileArgs& args) {
                    CompressorRef compressor(comp);
                    return saoProfile(compressor, args);
                });

        std::string kGenericNumericName = "numeric-ml-selector-64";
        mp[kGenericNumericName]         = std::make_shared<CompressProfile>(
                kGenericNumericName,
                profiles::getProfileDescription(kGenericNumericName),
                [](ZL_Compressor* comp, void*, const ProfileArgs& args) {
                    (void)args;
                    if (args.map().find("TBD") != args.map().end()) {
                        CompressorRef compressor(comp);
                        return unwrap(extractFolderOfCompressors(
                                compressor, args.map().at("TBD")));
                    } else {
                        return unwrap(
                                numeric64BitMLSelectorProfile(comp),
                                "Failed to set up numeric profile",
                                comp);
                    }
                });

        return mp;
    }();
    return staticProfiles;
};

} // namespace openzl::cli
