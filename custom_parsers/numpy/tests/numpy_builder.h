// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace openzl::custom_parsers::testing {

struct NpyArraySpec {
    /// A descr such as "<f4", quoted in the header, or a Python list for
    /// structured types, written as is.
    std::string descr;
    std::vector<uint64_t> shape;
    std::string data;
    bool fortranOrder = false;
};

/// The shape as Python writes a tuple: "()", "(3,)", "(3, 4)".
inline std::string shapeLiteral(const std::vector<uint64_t>& shape)
{
    std::string out = "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        out += (i > 0 ? ", " : "") + std::to_string(shape[i]);
    }
    return out + (shape.size() == 1 ? ",)" : ")");
}

/// The header dict as `np.save` writes it.
inline std::string headerDict(const NpyArraySpec& a)
{
    std::string descr = a.descr;
    if (!descr.starts_with("[")) {
        descr = "'" + descr + "'";
    }
    return "{'descr': " + descr
            + ", 'fortran_order': " + (a.fortranOrder ? "True" : "False")
            + ", 'shape': " + shapeLiteral(a.shape) + ", }";
}

/// A .npy file of format version @p major whose header is @p dict, padded
/// with spaces and a newline so that @p data starts at a multiple of 64.
inline std::string
buildNpyWithDict(std::string dict, std::string_view data, uint8_t major = 1)
{
    const size_t prefix = major == 1 ? 10 : 12;
    dict += std::string((64 - (prefix + dict.size() + 1) % 64) % 64, ' ');
    dict += '\n';
    std::string out(
            "\x93"
            "NUMPY");
    out += (char)major;
    out += '\0';
    for (size_t i = 0; i < prefix - 8; ++i) {
        out += (char)(dict.size() >> (8 * i));
    }
    return out + dict + std::string(data);
}

inline std::string buildNpy(const NpyArraySpec& a, uint8_t major = 1)
{
    return buildNpyWithDict(headerDict(a), a.data, major);
}

struct ZipMember {
    std::string name;
    /// The bytes as stored in the archive: with method 8 (deflate), they are
    /// not deflated here, as the lexer never inflates them.
    std::string data;
    uint16_t method = 0;
};

inline void le16(std::string& out, uint32_t v)
{
    out += (char)v;
    out += (char)(v >> 8);
}

inline void le32(std::string& out, uint32_t v)
{
    le16(out, v);
    le16(out, v >> 16);
}

/// A zip archive laid out as Python's zipfile writes it, with zero CRCs,
/// which the lexer does not check.
inline std::string buildZip(const std::vector<ZipMember>& members)
{
    std::string out;
    std::string central;
    for (const auto& m : members) {
        const auto offset = (uint32_t)out.size();
        le32(out, 0x04034b50);
        le16(out, 20); // version needed to extract
        le16(out, 0);  // flags
        le16(out, m.method);
        le32(out, 0); // time and date
        le32(out, 0); // CRC
        le32(out, (uint32_t)m.data.size());
        le32(out, (uint32_t)m.data.size());
        le16(out, (uint32_t)m.name.size());
        le16(out, 0); // extra field
        out += m.name + m.data;

        le32(central, 0x02014b50);
        le16(central, 20); // version made by
        le16(central, 20); // version needed to extract
        le16(central, 0);  // flags
        le16(central, m.method);
        le32(central, 0); // time and date
        le32(central, 0); // CRC
        le32(central, (uint32_t)m.data.size());
        le32(central, (uint32_t)m.data.size());
        le16(central, (uint32_t)m.name.size());
        le16(central, 0); // extra field
        le16(central, 0); // comment
        le16(central, 0); // disk
        le16(central, 0); // internal attributes
        le32(central, 0); // external attributes
        le32(central, offset);
        central += m.name;
    }
    const auto centralOffset = (uint32_t)out.size();
    out += central;
    // Counts of members that do not fit in 16 bits are in zip64 records
    const bool zip64 = members.size() > 0xFFFF;
    if (zip64) {
        const auto recordOffset = (uint32_t)out.size();
        le32(out, 0x06064b50);
        le32(out, 44); // size of the rest of the record, on 8 bytes
        le32(out, 0);
        le16(out, 45); // version made by
        le16(out, 45); // version needed to extract
        le32(out, 0);  // disk
        le32(out, 0);  // disk of the central directory
        for (int i = 0; i < 2; ++i) {
            le32(out, (uint32_t)members.size());
            le32(out, 0);
        }
        le32(out, (uint32_t)central.size());
        le32(out, 0);
        le32(out, centralOffset);
        le32(out, 0);

        le32(out, 0x07064b50);
        le32(out, 0); // disk of the zip64 record
        le32(out, recordOffset);
        le32(out, 0);
        le32(out, 1); // number of disks
    }
    const uint32_t count = zip64 ? 0xFFFF : (uint32_t)members.size();
    le32(out, 0x06054b50);
    le16(out, 0); // disk
    le16(out, 0); // disk of the central directory
    le16(out, count);
    le16(out, count);
    le32(out, (uint32_t)central.size());
    le32(out, centralOffset);
    le16(out, 0); // comment
    return out;
}

/// An archive as `np.savez` writes it: one stored member per array.
inline std::string buildNpz(
        const std::vector<std::pair<std::string, NpyArraySpec>>& arrays)
{
    std::vector<ZipMember> members;
    members.reserve(arrays.size());
    for (const auto& [name, array] : arrays) {
        members.push_back({ name + ".npy", buildNpy(array) });
    }
    return buildZip(members);
}

inline std::string pseudoRandomBytes(size_t size, uint32_t seed)
{
    std::string out(size, '\0');
    uint32_t state = seed * 2654435761u + 1;
    for (auto& c : out) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        c = (char)state;
    }
    return out;
}

} // namespace openzl::custom_parsers::testing
