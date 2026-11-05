// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace openzl::custom_parsers::testing {

/// An array, as NumPy pickles it.
struct PickledArray {
    std::string typeStr; // "f4", "u1", "U3"...
    std::vector<uint64_t> shape;
    std::string data;
    char byteOrder    = '<';
    bool fortranOrder = false;
    /// Names of the fields of a structured dtype.
    std::vector<std::string> fieldNames = {};
    /// Arrays of different ids have distinct, if equal, dtype objects, each
    /// pickled in full the first time.
    int dtypeId = 0;
};

/// Writes pickles as CPython does with protocols 2 to 5, or as Python 2 does
/// with protocol 2, memoizing objects in the same order.
class PickleWriter {
   public:
    explicit PickleWriter(int protocol, bool python2 = false)
            : protocol_(protocol), python2_(python2)
    {
    }

    void mark()
    {
        body_ += '(';
    }

    void none()
    {
        body_ += 'N';
    }

    void boolean(bool b)
    {
        body_ += b ? '\x88' : '\x89';
    }

    void integer(int64_t v)
    {
        if (v >= 0 && v <= 0xff) {
            body_ += 'K';
            le(v, 1);
        } else if (v >= 0 && v <= 0xffff) {
            body_ += 'M';
            le(v, 2);
        } else if (v >= INT32_MIN && v <= INT32_MAX) {
            body_ += 'J';
            le(v, 4);
        } else {
            body_ += '\x8a'; // LONG1
            body_ += '\x08';
            le(v, 8);
        }
    }

    /// CPython shares the strings of one character, which it then gets from
    /// the memo.
    void str(std::string_view s)
    {
        if (s.size() == 1 && getCached("s:" + std::string(s))) {
            return;
        }
        if (python2_) {
            binstring(s);
        } else if (protocol_ >= 4 && s.size() < 256) {
            body_ += '\x8c'; // SHORT_BINUNICODE
            le(s.size(), 1);
            body_ += s;
        } else {
            body_ += 'X'; // BINUNICODE
            le(s.size(), 4);
            body_ += s;
        }
        const uint64_t key = memoize();
        if (s.size() == 1) {
            cache_["s:" + std::string(s)] = key;
        }
    }

    void bytes(std::string_view b)
    {
        if (python2_) {
            binstring(b);
        } else if (protocol_ < 3) {
            encodedBytes(b);
        } else if (b.size() < 256) {
            body_ += 'C'; // SHORT_BINBYTES
            le(b.size(), 1);
            body_ += b;
        } else {
            body_ += 'B'; // BINBYTES
            le(b.size(), 4);
            body_ += b;
        }
        memoize();
    }

    void bytearray(std::string_view b)
    {
        body_ += '\x96'; // BYTEARRAY8
        le(b.size(), 8);
        body_ += b;
        memoize();
    }

    /// Writes the global, or gets it from the memo.
    void global(const std::string& module, const std::string& name)
    {
        if (getCached("g:" + module + "." + name)) {
            return;
        }
        if (protocol_ >= 4) {
            str(module);
            str(name);
            body_ += '\x93'; // STACK_GLOBAL
        } else {
            body_ += 'c' + module + '\n' + name + '\n';
        }
        cache_["g:" + module + "." + name] = memoize();
    }

    /// @returns the memo key of the object on top of the stack.
    uint64_t memoize()
    {
        const uint64_t key = memoCount_++;
        if (protocol_ >= 4) {
            body_ += '\x94'; // MEMOIZE
        } else if (key < 256) {
            body_ += 'q'; // BINPUT
            le(key, 1);
        } else {
            body_ += 'r'; // LONG_BINPUT
            le(key, 4);
        }
        return key;
    }

    void get(uint64_t key)
    {
        if (key < 256) {
            body_ += 'h'; // BINGET
            le(key, 1);
        } else {
            body_ += 'j'; // LONG_BINGET
            le(key, 4);
        }
    }

    /// Closes a tuple of @p n items, which follow a mark when @p n > 3.
    void tuple(size_t n)
    {
        if (n == 0) {
            body_ += ')';
            return;
        }
        body_ += n <= 3 ? (char)(0x84 + n) : 't';
        memoize();
    }

    void reduce()
    {
        body_ += 'R';
    }

    void build()
    {
        body_ += 'b';
    }

    void shape(const std::vector<uint64_t>& dims)
    {
        if (dims.size() > 3) {
            mark();
        }
        for (const auto d : dims) {
            integer((int64_t)d);
        }
        tuple(dims.size());
    }

    void dtype(const PickledArray& a)
    {
        std::string key =
                "d" + std::to_string(a.dtypeId) + ":" + a.byteOrder + a.typeStr;
        for (const auto& name : a.fieldNames) {
            key += "," + name;
        }
        if (getCached(key)) {
            return;
        }
        global("numpy", "dtype");
        str(a.typeStr);
        if (python2_) {
            integer(0);
            integer(1);
        } else {
            boolean(false);
            boolean(true);
        }
        tuple(3);
        reduce();
        cache_[key] = memoize();
        mark();
        integer(3);
        str(std::string(1, a.byteOrder));
        none();
        if (a.fieldNames.empty()) {
            none();
            none();
        } else {
            if (a.fieldNames.size() > 3) {
                mark();
            }
            for (const auto& name : a.fieldNames) {
                str(name);
            }
            tuple(a.fieldNames.size());
            body_ += '}'; // EMPTY_DICT, for the fields
            memoize();
        }
        integer(-1); // element size and alignment, which builtin types omit
        integer(-1);
        integer(0);
        tuple(8);
        build();
    }

    /// An array as ndarray.__reduce_ex__ pickles it: with _frombuffer from
    /// protocol 5, else with _reconstruct.
    void array(const PickledArray& a)
    {
        if (protocol_ >= 5) {
            global("numpy.core.numeric", "_frombuffer");
            mark();
            bytearray(a.data);
            dtype(a);
            shape(a.shape);
            str(a.fortranOrder ? "F" : "C");
            tuple(4);
            reduce();
            memoize();
            return;
        }
        global("numpy.core.multiarray", "_reconstruct");
        global("numpy", "ndarray");
        integer(0);
        tuple(1);
        bytes("b");
        tuple(3);
        reduce();
        memoize();
        mark();
        integer(1);
        shape(a.shape);
        dtype(a);
        boolean(a.fortranOrder);
        bytes(a.data);
        tuple(5);
        build();
    }

    void dictBegin()
    {
        body_ += '}';
        memoize();
        mark();
    }

    void dictEnd()
    {
        body_ += 'u'; // SETITEMS
    }

    void listBegin()
    {
        body_ += ']';
        memoize();
        mark();
    }

    void listEnd()
    {
        body_ += 'e'; // APPENDS
    }

    /// @returns the pickle, ended by STOP, in a single frame from protocol 4.
    std::string finish()
    {
        body_ += '.';
        std::string out = "\x80";
        out += (char)protocol_;
        if (protocol_ >= 4) {
            out += '\x95'; // FRAME
            for (size_t i = 0; i < 8; ++i) {
                out += (char)(body_.size() >> (8 * i));
            }
        }
        return out + body_;
    }

   private:
    void le(uint64_t v, size_t size)
    {
        for (size_t i = 0; i < size; ++i) {
            body_ += (char)(v >> (8 * i));
        }
    }

    void binstring(std::string_view s)
    {
        if (s.size() < 256) {
            body_ += 'U'; // SHORT_BINSTRING
            le(s.size(), 1);
        } else {
            body_ += 'T'; // BINSTRING
            le(s.size(), 4);
        }
        body_ += s;
    }

    /// Python 3 has no opcode for bytes below protocol 3: it calls
    /// _codecs.encode on their latin-1 decoding, which it writes as UTF-8.
    void encodedBytes(std::string_view b)
    {
        if (b.empty()) {
            global("__builtin__", "bytes");
            tuple(0);
            reduce();
            return;
        }
        global("_codecs", "encode");
        std::string text;
        for (const unsigned char c : b) {
            if (c < 0x80) {
                text += (char)c;
            } else {
                text += (char)(0xc0 | (c >> 6));
                text += (char)(0x80 | (c & 0x3f));
            }
        }
        str(text);
        if (!getCached("s:latin1")) {
            str("latin1");
            cache_["s:latin1"] = memoCount_ - 1;
        }
        tuple(2);
        reduce();
    }

    bool getCached(const std::string& key)
    {
        const auto it = cache_.find(key);
        if (it == cache_.end()) {
            return false;
        }
        get(it->second);
        return true;
    }

    int protocol_;
    bool python2_;
    std::string body_;
    uint64_t memoCount_ = 0;
    std::map<std::string, uint64_t> cache_;
};

/// A dict of arrays, as `pickle.dumps` writes it.
inline std::string pickleArrays(
        const std::vector<std::pair<std::string, PickledArray>>& arrays,
        int protocol,
        bool python2 = false)
{
    PickleWriter w(protocol, python2);
    w.dictBegin();
    for (const auto& [name, array] : arrays) {
        w.str(name);
        w.array(array);
    }
    w.dictEnd();
    return w.finish();
}

} // namespace openzl::custom_parsers::testing
