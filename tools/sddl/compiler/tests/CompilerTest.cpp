// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "tools/sddl/compiler/Compiler.h"
#include "tools/sddl/compiler/Exception.h"

using namespace testing;

namespace openzl::sddl::tests {

class CompilerTest : public Test {
   protected:
    void SetUp() override
    {
        compiler_ = std::make_unique<Compiler>(
                Compiler::Options{}.with_log(logs_).with_verbosity(verbosity_));
    }

    void expect_error(std::string_view source, std::string_view msg)
    {
        try {
            compiler_->compile(source, "[local_input]");
        } catch (const CompilerException& ex) {
            EXPECT_NE(std::string{ ex.what() }.find(msg), std::string::npos)
                    << std::quoted(ex.what()) << "\nShould contain:\n  "
                    << std::quoted(msg) << "\n"
                    << "Compiler debug logs:\n"
                    << logs_.str();
            return;
        }
        EXPECT_TRUE(false) << "Should have thrown a CompilerException!\n"
                           << "Compiler debug logs:\n"
                           << logs_.str();
    }

    void expect_success(std::string_view source)
    {
        EXPECT_NO_THROW(compiler_->compile(source, "[local_input]"))
                << "Compiler debug logs:\n"
                << logs_.str();
    }

    int verbosity_{ 3 };
    std::stringstream logs_;
    std::unique_ptr<Compiler> compiler_;
};

TEST_F(CompilerTest, ErrorMsgOpsMissingArgs)
{
    expect_error("foo = ;", "right-hand argument");
    expect_error("= foo;", "left-hand argument");
}

TEST_F(CompilerTest, ErrorMsgEmptyExpr)
{
    expect_error(";", "Empty expression");
}

TEST_F(CompilerTest, ErrorMsgNoOperatorBetweenSubExpressions)
{
    const auto prog = R"(
        tmp = 9 + 10 11 + 12
    )";
    expect_error(prog, "Expected operator between expressions");
}

TEST_F(CompilerTest, ErrorMsgTwoOperatorsBetweenSubExpressions)
{
    const auto prog = R"(
        tmp = 9 + 10 + + 11 + 12
    )";
    expect_error(prog, "Expected expression between operators");
}

TEST_F(CompilerTest, UnaryNegation)
{
    const auto prog = R"(
        tmp = 10 - - 11
    )";
    expect_success(prog);
}

TEST_F(CompilerTest, FieldAtEndOfFile)
{
    expect_success(": Byte");
}

TEST_F(CompilerTest, NonAsciiByteInSourceDoesNotCrash)
{
    // A non-ASCII byte anywhere in the source (comments included) is embedded
    // verbatim in the compiled output for debug info. Rendering that debug info
    // as JSON used to throw and abort the compiler because the JSON encoder
    // rejects bytes >= 0x80; compilation itself must succeed regardless of the
    // requested verbosity.
    const auto prog = std::string{ "# caf\xc3\xa9\n: Float32LE[_rem / 4]" };
    expect_success(prog);
}

TEST_F(CompilerTest, NonAsciiByteInSourceAtDefaultVerbosityDoesNotCrash)
{
    // Same as above, but with the default verbosity (0), which is what the
    // `sddl_compiler` CLI and the `zli --profile sddl` path use.
    const auto prog = std::string{ "# caf\xc3\xa9\n: Float32LE[_rem / 4]" };
    const Compiler compiler{};
    EXPECT_NO_THROW(compiler.compile(prog, "[local_input]"));
}

} // namespace openzl::sddl::tests
