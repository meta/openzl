// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <array>
#include <string_view>

#include "custom_parsers/zip_lexer.h"
#include "security/lionhead/utils/lib_ftest/ftest.h"

FUZZ(ZipLexerTest, FuzzLexer)
{
    std::string_view data{ (const char*)Data, Size };
    std::array<ZS2_ZipToken, 10> tokens;

    ZS2_ZipLexer lexer;
    auto report = ZS2_ZipLexer_init(&lexer, data.data(), data.size(), NULL);
    if (ZL_isError(report)) {
        return;
    }

    size_t offset = 0;
    while (!ZS2_ZipLexer_finished(&lexer)) {
        report = ZS2_ZipLexer_lex(&lexer, tokens.data(), tokens.size());
        if (ZL_isError(report)) {
            return;
        }
        const auto numTokens = ZL_validResult(report);
        ASSERT_LE(numTokens, tokens.size());
        for (size_t i = 0; i < numTokens; ++i) {
            ASSERT_EQ(tokens[i].ptr, data.data() + offset);
            offset += tokens[i].size;
        }
    }
    ASSERT_EQ(offset, data.size());
}
