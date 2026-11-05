// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <array>
#include <string>
#include <string_view>

#include "custom_parsers/pickle_lexer.h"
#include "security/lionhead/utils/lib_ftest/ftest.h"

/// Tokens cover the input in order, whatever it holds. Inputs start with the
/// PROTO opcode, to go past the first opcode.
FUZZ(PickleLexerTest, FuzzLexer)
{
    const std::string data =
            std::string("\x80\x04", 2) + std::string((const char*)Data, Size);
    std::array<ZL_PickleToken, 4> tokens{};

    ZL_PickleLexer lexer;
    ZL_PickleLexer_init(&lexer, data.data(), data.size());
    size_t offset = 0;
    while (!ZL_PickleLexer_finished(&lexer)) {
        const size_t numTokens =
                ZL_PickleLexer_lex(&lexer, tokens.data(), tokens.size());
        ASSERT_LE(numTokens, tokens.size());
        for (size_t i = 0; i < numTokens; ++i) {
            ASSERT_EQ(tokens[i].ptr, data.data() + offset);
            ASSERT_GT(tokens[i].size, 0u);
            offset += tokens[i].size;
        }
    }
    ASSERT_EQ(offset, data.size());
}
