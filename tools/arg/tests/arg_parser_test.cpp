// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "tools/arg/ParseException.h"
#include "tools/arg/arg_parser.h"

using namespace openzl::arg;

namespace {

// Test that --flag=value syntax works correctly
TEST(ArgParserTest, LongFlagWithEqualsSign)
{
    ArgParser parser;
    parser.addCommand(1, "test", 't');
    parser.addCommandFlag(1, "profile", 'p', true, "Profile name");

    const char* argv[] = { "prog", "test", "--profile=u32" };
    auto parsed        = parser.parse(3, const_cast<char**>(argv));

    EXPECT_EQ(parsed.cmdFlag(1, "profile").value(), "u32");
}

// Test that flag value with no = works correctly
TEST(ArgParserTest, LongFlagWithSpaceSeparatedValue)
{
    ArgParser parser;
    parser.addCommand(1, "test", 't');
    parser.addCommandFlag(1, "profile", 'p', true, "Profile name");

    const char* argv[] = { "prog", "test", "--profile", "u32" };
    auto parsed        = parser.parse(4, const_cast<char**>(argv));

    EXPECT_EQ(parsed.cmdFlag(1, "profile").value(), "u32");
}

// Test that both syntaxes produce the same result
TEST(ArgParserTest, BothSyntaxesProduceSameResult)
{
    ArgParser parser1;
    parser1.addCommand(1, "test", 't');
    parser1.addCommandFlag(1, "profile", 'p', true, "Profile name");

    const char* argv1[] = { "prog", "test", "--profile=u32" };
    auto parsed1        = parser1.parse(3, const_cast<char**>(argv1));

    ArgParser parser2;
    parser2.addCommand(1, "test", 't');
    parser2.addCommandFlag(1, "profile", 'p', true, "Profile name");

    const char* argv2[] = { "prog", "test", "--profile", "u32" };
    auto parsed2        = parser2.parse(4, const_cast<char**>(argv2));

    EXPECT_EQ(
            parsed1.cmdFlag(1, "profile").value(),
            parsed2.cmdFlag(1, "profile").value());
}

// Test Equal Syntax with multiple values
TEST(ArgParserTest, EqualSyntaxWithMultipleValues)
{
    ArgParser parser;
    parser.addCommand(1, "test", 't');
    parser.addCommandFlag(1, "profile", 'p', true, "Profile name");
    parser.addCommandFlag(1, "output", 'o', true, "Output path");

    const char* argv[] = {
        "prog", "test", "--profile=u32", "--output=file.zl"
    };
    auto parsed = parser.parse(4, const_cast<char**>(argv));

    EXPECT_EQ(parsed.cmdFlag(1, "profile").value(), "u32");
    EXPECT_EQ(parsed.cmdFlag(1, "output").value(), "file.zl");
}

// Test mixed syntax (Some with =, some without)
TEST(ArgParserTest, MixedSyntax)
{
    ArgParser parser;
    parser.addCommand(1, "test", 't');
    parser.addCommandFlag(1, "profile", 'p', true, "Profile name");
    parser.addCommandFlag(1, "output", 'o', true, "Output path");

    const char* argv[] = {
        "prog", "test", "--profile=u32", "--output", "file.zl"
    };

    auto parsed = parser.parse(5, const_cast<char**>(argv));

    EXPECT_EQ(parsed.cmdFlag(1, "profile").value(), "u32");
    EXPECT_EQ(parsed.cmdFlag(1, "output").value(), "file.zl");
}

// Test that boolean flag (no value) works correctly
TEST(ArgParserTest, BooleanFlagWithoutValue)
{
    ArgParser parser;
    parser.addCommand(1, "test", 't');
    parser.addCommandFlag(1, "force", 'f', false, "Force overwrite");

    const char* argv[] = { "prog", "test", "--force" };
    auto parsed        = parser.parse(3, const_cast<char**>(argv));

    EXPECT_TRUE(parsed.cmdFlag(1, "force"));
}

// Test that value containing special characters (like paths) works with =
// syntax
TEST(ArgParserTest, ValueContainingSpecialCharacters)
{
    ArgParser parser;
    parser.addCommand(1, "test", 't');
    parser.addCommandFlag(1, "path", 'p', true, "Path Value"); // hasVal=true

    const char* argv[] = { "prog", "test", "--path=/tmp/test-file.txt" };
    auto parsed        = parser.parse(3, const_cast<char**>(argv));

    EXPECT_EQ(parsed.cmdFlag(1, "path").value(), "/tmp/test-file.txt");
}

// Test that boolean flag with = value gives proper error
TEST(ArgParserTest, BooleanFlagWithEqualsValueThrows)
{
    ArgParser parser;
    parser.addCommand(1, "test", 't');
    parser.addCommandFlag(1, "force", 'f', false, "Force overwrite");

    const char* argv[] = { "prog", "test", "--force=true" };

    EXPECT_THROW(
            {
                try {
                    parser.parse(3, const_cast<char**>(argv));
                } catch (const ParseException& e) {
                    std::string errorMsg = e.what();
                    EXPECT_NE(
                            errorMsg.find("does not accept a value"),
                            std::string::npos);
                    throw;
                }
            },
            ParseException);
}

// Test that unknown flag with equals sign gives proper error
TEST(ArgParserTest, UnknownFlagWithEquals)
{
    ArgParser parser;
    parser.addCommand(1, "test", 't');
    parser.addCommandFlag(1, "profile", 'p', true, "Profile name");

    const char* argv[] = { "prog", "test", "--unknown=value" };

    EXPECT_THROW(
            {
                try {
                    parser.parse(3, const_cast<char**>(argv));
                } catch (const ParseException& e) {
                    std::string errorMsg = e.what();
                    EXPECT_NE(
                            errorMsg.find("Unknown option: --unknown"),
                            std::string::npos);
                    throw;
                }
            },
            ParseException);
}

TEST(ArgParserTest, EqualsWithEmptyValueThrows)
{
    ArgParser parser;
    parser.addCommand(1, "test", 't');
    parser.addCommandFlag(1, "profile", 'p', true, "Profile name");

    const char* argv[] = { "prog", "test", "--profile=" };

    EXPECT_THROW(
            {
                try {
                    parser.parse(3, const_cast<char**>(argv));
                } catch (const ParseException& e) {
                    std::string errorMsg = e.what();
                    EXPECT_NE(
                            errorMsg.find("requires a value"),
                            std::string::npos);
                    throw;
                }
            },
            ParseException);
}

// Test that arguments after -- are positionals, even when they are spelled
// like a subcommand
TEST(ArgParserTest, SubcommandNameAfterEndOfOptionsIsPositional)
{
    ArgParser parser;
    parser.addCommand(1, "test", 't');
    parser.addCommandPositional(1, "input", "Input file path");

    for (const char* name : { "t", "test" }) {
        const char* argv[] = { "prog", "test", "--", name };
        auto parsed        = parser.parse(4, const_cast<char**>(argv));
        parser.validate(parsed);

        EXPECT_EQ(parsed.chosenCmd(), 1);
        EXPECT_EQ(parsed.cmdPositional(1, "input"), name);
    }
}

// Test that -- does not prevent choosing the subcommand
TEST(ArgParserTest, SubcommandAfterEndOfOptions)
{
    ArgParser parser;
    parser.addCommand(1, "test", 't');
    parser.addCommandPositional(1, "input", "Input file path");

    const char* argv[] = { "prog", "--", "test", "file" };
    auto parsed        = parser.parse(4, const_cast<char**>(argv));
    parser.validate(parsed);

    EXPECT_EQ(parsed.chosenCmd(), 1);
    EXPECT_EQ(parsed.cmdPositional(1, "input"), "file");
}

} // namespace
