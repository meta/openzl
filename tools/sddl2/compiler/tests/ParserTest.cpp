// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "tools/sddl2/compiler/tests/CompilerTest.h"

namespace openzl::sddl2::tests {

namespace {
class ParserTest : public CompilerTest {};
} // namespace

TEST_F(ParserTest, ErrorMsgOpsMissingArgs)
{
    expect_error("foo = \n", "Rule requires 1 rhs args but 0 found.");
    expect_error("= foo\n", "Rule requires a lhs arg but none found.");
}

TEST_F(ParserTest, ErrorMsgCallMissingCallee)
{
    expect_error("()", "Rule requires a lhs arg but none found.");
    expect_error("()\n", "Rule requires a lhs arg but none found.");
}

TEST_F(ParserTest, ErrorMsgNoOperatorBetweenSubExpressions)
{
    const auto prog = R"(
        tmp = 9 + 10 11 + 12
    )";
    expect_error(prog, "Expected operator between expressions");
}

TEST_F(ParserTest, ErrorMsgTwoOperatorsBetweenSubExpressions)
{
    const auto prog = R"(
        tmp = 9 + 10 + + 11 + 12
    )";
    expect_error(prog, "Failed to match args for rule");
}

TEST_F(ParserTest, ParseSimpleOps)
{
    const auto prog = R"(
        # arithmetic operators
        expect 1 + 2
        expect 1 - 2
        expect 1 * 2
        expect 1 / 2
        expect 1 % 2

        # conditional operators
        expect 1 < 2
        expect 1 <= 2
        expect 1 > 2 == 0
        expect 1 >= 2 == 0
        expect 2 == 2
        expect 1 != 2

        # logical operators
        expect 1 && 2
        expect 1 || 2
        expect !0

        # bitwise operators
        expect 1 & 2
        expect 1 | 2
        expect 1 ^ 2
        expect ~1
    )";
    expect_success(prog);
}

TEST_F(ParserTest, ConsumeBuiltinFieldsAST)
{
    const auto prog     = R"(
        # integer numeric types
        : Byte
        : UInt8
        : Int8
        : UInt16LE
        : UInt16BE
        : Int16LE
        : Int16BE
        : UInt32LE
        : UInt32BE
        : Int32LE
        : Int32BE
        : UInt64LE
        : UInt64BE
        : Int64LE
        : Int64BE

        # float numeric types
        : Float16LE
        : Float16BE
        : Float32LE
        : Float32BE
        : Float64LE
        : Float64BE
        : BFloat16LE
        : BFloat16BE
    )";
    const auto cg       = Codegen();
    const auto expected = std::vector<ASTPtr>({
            cg.consume(cg.builtin_field(Symbol::BYTE)),
            cg.consume(cg.builtin_field(Symbol::U8)),
            cg.consume(cg.builtin_field(Symbol::I8)),
            cg.consume(cg.builtin_field(Symbol::U16LE)),
            cg.consume(cg.builtin_field(Symbol::U16BE)),
            cg.consume(cg.builtin_field(Symbol::I16LE)),
            cg.consume(cg.builtin_field(Symbol::I16BE)),
            cg.consume(cg.builtin_field(Symbol::U32LE)),
            cg.consume(cg.builtin_field(Symbol::U32BE)),
            cg.consume(cg.builtin_field(Symbol::I32LE)),
            cg.consume(cg.builtin_field(Symbol::I32BE)),
            cg.consume(cg.builtin_field(Symbol::U64LE)),
            cg.consume(cg.builtin_field(Symbol::U64BE)),
            cg.consume(cg.builtin_field(Symbol::I64LE)),
            cg.consume(cg.builtin_field(Symbol::I64BE)),
            cg.consume(cg.builtin_field(Symbol::F16LE)),
            cg.consume(cg.builtin_field(Symbol::F16BE)),
            cg.consume(cg.builtin_field(Symbol::F32LE)),
            cg.consume(cg.builtin_field(Symbol::F32BE)),
            cg.consume(cg.builtin_field(Symbol::F64LE)),
            cg.consume(cg.builtin_field(Symbol::F64BE)),
            cg.consume(cg.builtin_field(Symbol::BF16LE)),
            cg.consume(cg.builtin_field(Symbol::BF16BE)),
    });
    expect_ast(prog, expected);
}

TEST_F(ParserTest, UnaryNegationAST)
{
    const auto prog = R"(
        tmp = 10 - - 11
        expect tmp == 21
    )";

    const auto cg       = Codegen();
    const auto expected = std::vector<ASTPtr>({
            cg.assign(cg.var("tmp"), cg.sub(cg.num(10), cg.neg(cg.num(11)))),
            cg.expect(cg.eq(cg.var("tmp"), cg.num(21))),
    });
    expect_ast(prog, expected);
}

TEST_F(ParserTest, SimpleArithmeticAST)
{
    const auto prog = R"(
       expect 1 + 2 == 3
    )";

    const auto cg       = Codegen();
    const auto expected = std::vector<ASTPtr>({
            cg.expect(cg.eq(cg.add(cg.num(1), cg.num(2)), cg.num(3))),
    });
    expect_ast(prog, expected);
}

TEST_F(ParserTest, ArrayAST)
{
    const auto prog = R"(
        len =  1 + 2
        : Byte[len]
        : Byte[]
    )";

    const auto cg       = Codegen();
    const auto expected = std::vector<ASTPtr>(
            { cg.assign(cg.var("len"), cg.add(cg.num(1), cg.num(2))),
              cg.consume(
                      cg.array(cg.builtin_field(Symbol::BYTE), cg.var("len"))),
              cg.consume(cg.array(cg.builtin_field(Symbol::BYTE))) });

    expect_ast(prog, expected);
}

TEST_F(ParserTest, ArrayAtEndOfFileAST)
{
    const auto cg = Codegen();
    for (const auto suffix : { "", " ", " # comment", "\n" }) {
        expect_ast(
                std::string(": Byte[]") + suffix,
                { cg.consume(cg.array(cg.builtin_field(Symbol::BYTE))) });
    }
}

TEST_F(ParserTest, RecordAST)
{
    const auto prog = R"(
        record Entry() {
            id: Int32LE,
        }
        : Entry
    )";

    const auto cg       = Codegen();
    const auto expected = std::vector<ASTPtr>(
            { cg.assign(
                      cg.var("Entry"),
                      cg.record(
                              ArgVec{},
                              ArgVec{ cg.assume(
                                      cg.var("id"),
                                      cg.builtin_field(Symbol::I32LE)) })),
              cg.consume(cg.var("Entry")) });
    expect_ast(prog, expected);
}

TEST_F(ParserTest, AnonymousRecordAST)
{
    const auto prog = R"(
        : record() {id: Int32LE, val: Int32LE}
    )";

    const auto cg       = Codegen();
    const auto expected = std::vector<ASTPtr>({ cg.consume(cg.record(
            ArgVec{},
            ArgVec{ cg.assume(cg.var("id"), cg.builtin_field(Symbol::I32LE)),
                    cg.assume(
                            cg.var("val"),
                            cg.builtin_field(Symbol::I32LE)) })) });
    expect_ast(prog, expected);
}

TEST_F(ParserTest, ParenthesesOverridePrecedenceAST)
{
    const auto prog = R"(
        tmp = (1 - 2) * 3
        expect tmp == -3
    )";

    const auto cg       = Codegen();
    const auto expected = std::vector<ASTPtr>({
            cg.assign(
                    cg.var("tmp"),
                    cg.mul(cg.sub(cg.num(1), cg.num(2)), cg.num(3))),
            cg.expect(cg.eq(cg.var("tmp"), cg.neg(cg.num(3)))),
    });
    expect_ast(prog, expected);
}

TEST_F(ParserTest, NestedParenthesesAST)
{
    const auto prog = R"(
        tmp = ((1 + 2) * (3 + 4))
        expect tmp == 21
    )";

    const auto cg       = Codegen();
    const auto expected = std::vector<ASTPtr>({
            cg.assign(
                    cg.var("tmp"),
                    cg.mul(cg.add(cg.num(1), cg.num(2)),
                           cg.add(cg.num(3), cg.num(4)))),
            cg.expect(cg.eq(cg.var("tmp"), cg.num(21))),
    });
    expect_ast(prog, expected);
}

TEST_F(ParserTest, ComplexArithmeticExpressionAST)
{
    const auto prog = R"(
        tmp = 1 + 2 * 3 - 4 / 2
        expect tmp == 5
    )";

    const auto cg       = Codegen();
    const auto expected = std::vector<ASTPtr>({
            cg.assign(
                    cg.var("tmp"),
                    cg.sub(cg.add(cg.num(1), cg.mul(cg.num(2), cg.num(3))),
                           cg.div(cg.num(4), cg.num(2)))),
            cg.expect(cg.eq(cg.var("tmp"), cg.num(5))),
    });
    expect_ast(prog, expected);
}

TEST_F(ParserTest, SimpleSaoAST)
{
    const auto prog = R"(
        # Star catalog entry (28 bytes)
        record StarEntry() {
            SRA0:  Float64LE,    # Right Ascension (radians)
            SDEC0: Float64LE,    # Declination (radians)
            ISP:   Bytes(2),     # Spectral type
            MAG:   Int16LE,      # Magnitude
            XRPM:  Float32LE,    # R.A. proper motion
            XDPM:  Float32LE     # Dec. proper motion
        }

        # File structure
        header: Bytes(28)
        stars: StarEntry[10]
    )";

    // TODO: Update this test once auto-sized arrays are supported
    const auto cg       = Codegen();
    const auto expected = std::vector<ASTPtr>({
            cg.assign(
                    cg.var("StarEntry"),
                    cg.record(
                            ArgVec{},
                            ArgVec{
                                    cg.assume(
                                            cg.var("SRA0"),
                                            cg.builtin_field(Symbol::F64LE)),
                                    cg.assume(
                                            cg.var("SDEC0"),
                                            cg.builtin_field(Symbol::F64LE)),
                                    cg.assume(
                                            cg.var("ISP"), cg.bytes(cg.num(2))),
                                    cg.assume(
                                            cg.var("MAG"),
                                            cg.builtin_field(Symbol::I16LE)),
                                    cg.assume(
                                            cg.var("XRPM"),
                                            cg.builtin_field(Symbol::F32LE)),
                                    cg.assume(
                                            cg.var("XDPM"),
                                            cg.builtin_field(Symbol::F32LE)),
                            })),
            cg.assume(cg.var("header"), cg.bytes(cg.num(28))),
            cg.assume(
                    cg.var("stars"), cg.array(cg.var("StarEntry"), cg.num(10))),
    });
    expect_ast(prog, expected);
}

TEST_F(ParserTest, CallAST)
{
    const auto prog = R"(
        record Foo(A, B) {
            x: Int32LE[A],
            y: Int16LE[B]
        }
        : Foo(3, 5)
    )";

    const auto cg       = Codegen();
    const auto expected = std::vector<ASTPtr>({
            cg.assign(
                    cg.var("Foo"),
                    cg.record(
                            ArgVec{ cg.var("A"), cg.var("B") },
                            ArgVec{
                                    cg.assume(
                                            cg.var("x"),
                                            cg.array(
                                                    cg.builtin_field(
                                                            Symbol::I32LE),
                                                    cg.var("A"))),
                                    cg.assume(
                                            cg.var("y"),
                                            cg.array(
                                                    cg.builtin_field(
                                                            Symbol::I16LE),
                                                    cg.var("B"))),
                            })),
            cg.consume(cg.call(cg.var("Foo"), ArgVec{ cg.num(3), cg.num(5) })),
    });
    expect_ast(prog, expected);
}

TEST_F(ParserTest, WhenBlockAST)
{
    const auto prog = R"(
        flag: UInt8
        when flag == 1 {
            x: Int32LE
        }
    )";

    const auto cg       = Codegen();
    const auto expected = std::vector<ASTPtr>({
            cg.assume(cg.var("flag"), cg.builtin_field(Symbol::U8)),
            cg.when(cg.eq(cg.var("flag"), cg.num(1)),
                    ArgVec{ cg.assume(
                            cg.var("x"), cg.builtin_field(Symbol::I32LE)) }),
    });
    expect_ast(prog, expected);
}

TEST_F(ParserTest, BetweenAST)
{
    const auto prog = R"(
        expect between(1, 2, 3)
    )";

    const auto cg       = Codegen();
    const auto expected = std::vector<ASTPtr>({
            cg.expect(cg.between(cg.num(1), cg.num(2), cg.num(3))),
    });
    expect_ast(prog, expected);
}

TEST_F(ParserTest, BetweenWrongArity)
{
    expect_error(
            "expect between(1, 2)\n", "between() requires exactly 3 arguments");
    expect_error(
            "expect between(1, 2, 3, 4)\n",
            "between() requires exactly 3 arguments");
}

TEST_F(ParserTest, WhenBlockInRecordAST)
{
    const auto prog = R"(
        record Data(flag) {
            base: UInt32LE,
            when flag {
                optional: UInt16LE
            }
        }
        : Data(1)
    )";

    const auto cg       = Codegen();
    const auto expected = std::vector<ASTPtr>({
            cg.assign(
                    cg.var("Data"),
                    cg.record(
                            ArgVec{ cg.var("flag") },
                            ArgVec{
                                    cg.assume(
                                            cg.var("base"),
                                            cg.builtin_field(Symbol::U32LE)),
                                    cg.when(cg.var("flag"),
                                            ArgVec{ cg.assume(
                                                    cg.var("optional"),
                                                    cg.builtin_field(
                                                            Symbol::U16LE)) }),
                            })),
            cg.consume(cg.call(cg.var("Data"), ArgVec{ cg.num(1) })),
    });
    expect_ast(prog, expected);
}

TEST_F(ParserTest, AnnotationOnNonRecord)
{
    const auto prog = R"(
        x = 5 @some_annotation
    )";
    expect_error(prog, "can only be applied to a record");
}

TEST_F(ParserTest, AnnotationNameRequiresIdentifier)
{
    const auto prog = R"(
        record Foo() {
            x: Int32LE
        } @ 5
    )";
    expect_error(prog, "Annotation name must be an identifier");
}

TEST_F(ParserTest, AnnotationAllowsSpaceAfterSigil)
{
    // `@ name` (with a space) must tokenize and parse the same as `@name`.
    const auto prog = R"(
        record Foo() {
            x: Int32LE
        } @ some_annotation
    )";
    EXPECT_NO_THROW(compiler_->compile_ast(prog, "[local_input]"))
            << "Compiler debug logs:\n"
            << logs_.str();
}

} // namespace openzl::sddl2::tests
