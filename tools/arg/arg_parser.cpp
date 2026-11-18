// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "tools/arg/arg_parser.h"

#include <algorithm>
#include <cstring>
#include <sstream>

#include "tools/arg/ParseException.h"

namespace openzl::arg {

namespace {

bool isShortOpt(const char* str)
{
    return str[0] == '-' && str[1] != '-' && str[1] != '\0';
}

bool isLongOpt(const char* str)
{
    return str[0] == '-' && str[1] == '-';
}

std::optional<const Flag*> findFlagByLong(
        const char* longName,
        const std::vector<Flag>& flags)
{
    auto iter = std::find_if(
            flags.begin(), flags.end(), [longName](const Flag& flag) {
                return flag.name == longName;
            });
    return (iter == flags.end()) ? std::nullopt
                                 : std::optional<const Flag*>(&*iter);
}

std::optional<const Flag*> findFlagByShort(
        char shortName,
        const std::vector<Flag>& flags)
{
    auto iter = std::find_if(
            flags.begin(), flags.end(), [shortName](const Flag& flag) {
                return flag.shortName == shortName;
            });
    return (iter == flags.end()) ? std::nullopt
                                 : std::optional<const Flag*>(&*iter);
}

} // namespace

ArgParser::ArgParser()
{
    cmdFlags_[CMD_UNSPECIFIED] = {};
}

void ArgParser::addGlobalFlag(
        const std::string& name,
        char shortName,
        bool hasVal,
        const std::string& help)
{
    addCommandFlag(CMD_UNSPECIFIED, name, shortName, hasVal, help);
}

void ArgParser::addGlobalImmediate(
        const std::string& name,
        char shortName,
        bool hasVal,
        const std::string& help)
{
    addCommandImmediate(CMD_UNSPECIFIED, name, shortName, hasVal, help);
}

void ArgParser::addCommand(int cmd, const std::string& name, char shortName)
{
    commands_.push_back({ cmd, name, shortName });
    cmdPositionals_[cmd] = {};
}

void ArgParser::addCommandFlag(
        int cmd,
        const std::string& name,
        char shortName,
        bool hasVal,
        const std::string& help)
{
    cmdFlags_[cmd].push_back({ name, shortName, false, hasVal, help });
}

void ArgParser::addCommandImmediate(
        int cmd,
        const std::string& name,
        char shortName,
        bool hasVal,
        const std::string& help)
{
    cmdFlags_[cmd].push_back({ name, shortName, true, hasVal, help });
}

void ArgParser::addCommandPositional(
        int cmd,
        NumArgs numArgs,
        const std::string& name,
        const std::string& help)
{
    if (!cmdPositionals_[cmd].empty()
        && cmdPositionals_[cmd].back().numArgs != NumArgs::One) {
        throw ParseException(
                "Cannot add a positional argument after a variadic positional argument!");
    }
    cmdPositionals_[cmd].push_back({ numArgs, name, help });
}

std::string ArgParser::help() const
{
    std::stringstream ss;
    if (cmdFlags_.find(CMD_UNSPECIFIED) != cmdFlags_.end()
        && !cmdFlags_.at(CMD_UNSPECIFIED).empty()) {
        ss << "Global Options:\n";
        for (const auto& flag : cmdFlags_.at(CMD_UNSPECIFIED)) {
            ss << "  --" << flag.name;
            if (flag.shortName != 0) {
                ss << ", -" << flag.shortName;
            }
            ss << "\n";
            ss << "    " << flag.help << "\n";
        }
        ss << "\n";
    }

    ss << "Commands:\n";
    ss << "  ";
    for (const auto& command : commands_) {
        ss << " " << command.name;
    }
    ss << "\n";
    for (const auto& command : commands_) {
        ss << "=====\n";
        ss << help(command) << "\n";
    }
    return ss.str();
}

std::string ArgParser::help(int cmd) const
{
    std::stringstream ss;
    if (cmdFlags_.find(CMD_UNSPECIFIED) != cmdFlags_.end()
        && !cmdFlags_.at(CMD_UNSPECIFIED).empty()) {
        ss << "Global Options:\n";
        for (const auto& flag : cmdFlags_.at(CMD_UNSPECIFIED)) {
            ss << "  --" << flag.name;
            if (flag.shortName != 0) {
                ss << ", -" << flag.shortName;
            }
            ss << "\n";
            ss << "    " << flag.help << "\n";
        }
        ss << "\n";
    }

    ss << "Command '" << commandName(cmd) << "' Options :\n";
    for (const auto& command : commands_) {
        if (command.cmd == cmd) {
            ss << help(command);
        }
    }
    return ss.str();
}

std::string ArgParser::help(const Command& command) const
{
    std::stringstream ss;
    ss << command.name << ", " << command.shortName << ":\n";
    ss << "Positionals:\n";
    if (cmdPositionals_.find(command.cmd) != cmdPositionals_.end()) {
        for (auto const& positional : cmdPositionals_.at(command.cmd)) {
            switch (positional.numArgs) {
                case NumArgs::ZeroOrOne:
                    ss << "  [" << positional.name << "]\n";
                    break;
                case NumArgs::One:
                    ss << "  " << positional.name << "\n";
                    break;
                case NumArgs::OneOrMore:
                    ss << "  " << positional.name << "...\n";
                    break;
                case NumArgs::ZeroOrMore:
                    ss << "  [" << positional.name << "]...\n";
                    break;
            }
            ss << "    " << positional.help << "\n";
        }
    }
    ss << "Options:\n";
    if (cmdFlags_.find(command.cmd) != cmdFlags_.end()) {
        for (const auto& flag : cmdFlags_.at(command.cmd)) {
            ss << "  --" << flag.name;
            if (flag.shortName != 0) {
                ss << ", -" << flag.shortName;
            }
            ss << "\n";
            ss << "    " << flag.help << "\n";
        }
    }
    return ss.str();
}

std::string ArgParser::commandName(int cmd) const
{
    for (const auto& command : commands_) {
        if (command.cmd == cmd) {
            return command.name;
        }
    }
    return std::to_string(cmd);
}

ParsedArgs ArgParser::parse(int argc, char** argv) const
{
    ParsedArgs parsedArgs;
    parsedArgs.cmdFlags_ = cmdFlags_; // copy flags database into the parsedArgs
                                      // object in case the ArgParser goes out
                                      // of scope before the ParsedArgs object
    std::map<int, size_t> cmdPositionalIndex;
    bool seenEndOfOptions = false;

    for (int i = 1; i < argc; ++i) {
        auto findFlag = [&](bool isShort, std::string& str, int chosenCmd)
                -> std::optional<std::pair<const Flag*, int>> {
            std::optional<const Flag*> ptr;
            // search global flags first
            if (isShort) {
                ptr = findFlagByShort(str[0], cmdFlags_.at(CMD_UNSPECIFIED));
            } else {
                ptr = findFlagByLong(
                        str.c_str(), cmdFlags_.at(CMD_UNSPECIFIED));
            }
            if (ptr) {
                return std::optional<std::pair<const Flag*, int>>(
                        std::make_pair(ptr.value(), CMD_UNSPECIFIED));
            }

            // otherwise, search command-specific flags
            if (isShort) {
                ptr = findFlagByShort(str[0], cmdFlags_.at(chosenCmd));
            } else {
                ptr = findFlagByLong(str.c_str(), cmdFlags_.at(chosenCmd));
            }
            return (ptr) ? std::optional<std::pair<const Flag*, int>>(
                                   { ptr.value(), chosenCmd })
                         : std::nullopt;
        };
        auto formatFlagName = [](const Flag* flag) {
            auto flagName = "--" + std::string(flag->name);
            if (flag->shortName != 0) {
                flagName += ", -" + std::string(1, flag->shortName);
            }
            return flagName;
        };
        auto checkDuplicateFlag = [&](const Flag* flag, int cmd) {
            if (parsedArgs.cmdVals_[cmd].find(flag->name)
                != parsedArgs.cmdVals_[cmd].end()) {
                throw ParseException(
                        "Option " + formatFlagName(flag)
                        + " specified more than once");
            }
        };
        auto populateOption = [&](const Flag* flag, int cmd) {
            checkDuplicateFlag(flag, cmd);

            if (flag->hasVal) {
                ++i;
                if (i >= argc || isShortOpt(argv[i]) || isLongOpt(argv[i])) {
                    throw ParseException(
                            "Option " + formatFlagName(flag)
                            + " requires a value");
                }
                parsedArgs.cmdVals_[cmd][flag->name].push_back(argv[i]);
            } else {
                parsedArgs.cmdVals_[cmd][flag->name].push_back("");
            }
        };
        auto processShortFlag = [&](char shortName, int chosenCmd) {
            auto str     = std::string{ shortName };
            auto flagOpt = findFlag(true, str, chosenCmd);
            if (!flagOpt.has_value()) {
                std::string err = "Unknown option: " + std::string(argv[i]);
                throw ParseException(err);
            }
            auto [flag, cmd] = flagOpt.value();
            populateOption(flag, cmd);
        };
        auto processLongFlag = [&](char* name, int chosenCmd) {
            std::string str{ name };
            auto equalPos = str.find('=');

            auto flagName = (equalPos != std::string::npos)
                    ? str.substr(0, equalPos)
                    : str;

            auto flagOpt = findFlag(false, flagName, chosenCmd);
            if (!flagOpt.has_value()) {
                throw ParseException("Unknown option: --" + flagName);
            }
            auto [flag, cmd] = flagOpt.value();

            // No = syntax, use populateOption for space-separated or boolean
            if (equalPos == std::string::npos) {
                populateOption(flag, cmd);
                return;
            }

            auto flagValue = str.substr(equalPos + 1);

            if (!flag->hasVal) {
                throw ParseException(
                        "Option " + formatFlagName(flag)
                        + " does not accept a value, but got '=" + flagValue
                        + "'");
            }
            if (flagValue.empty()) {
                throw ParseException(
                        "Option " + formatFlagName(flag)
                        + " requires a value, but got '--" + flag->name + "='");
            }

            checkDuplicateFlag(flag, cmd);
            parsedArgs.cmdVals_[cmd][flag->name].push_back(flagValue);
        };

        if (seenEndOfOptions) {
            goto POSITIONAL;
        }

        if (isShortOpt(argv[i])) {
            if (argv[i][1] == 'v'
                && (strlen(argv[i]) > 2
                    || (strlen(argv[i]) == 2
                        && (i + 1 >= argc || !std::isdigit(argv[i + 1][0]))))) {
                // special case for verbosity (-v -> 4, -vv -> 5, etc)
                std::string arg = argv[i];
                for (size_t j = 1; j < arg.size(); ++j) {
                    if (arg[j] != 'v') {
                        std::string err = "Unknown short option: " + arg;
                        throw ParseException(err);
                    }
                }
                std::string verboseFlag = "v";
                auto flagOpt =
                        findFlag(true, verboseFlag, parsedArgs.chosenCmd_);
                if (!flagOpt.has_value()) {
                    std::string err =
                            "Unable to find matching verbosity argument for option: "
                            + arg;
                    throw ParseException(err);
                }
                auto [flag, cmd] = flagOpt.value();
                parsedArgs.cmdVals_[cmd][flag->name].push_back(
                        std::to_string(arg.size() + 2));
            } else {
                processShortFlag(argv[i][1], parsedArgs.chosenCmd_);
            }
        } else if (isLongOpt(argv[i])) {
            if (argv[i][2] == 0) {
                // '--', signals end of options
                seenEndOfOptions = true;
                continue;
            }
            processLongFlag(argv[i] + 2, parsedArgs.chosenCmd_);
        } else {
        POSITIONAL:
            // positional argument, is either command-specific positional or
            // subcommand string. After '--', it is always a positional of the
            // chosen subcommand, even if it is spelled like a subcommand.
            const bool mayBeSubcommand = !seenEndOfOptions
                    || parsedArgs.chosenCmd_ == CMD_UNSPECIFIED;
            bool isSubcommand = false;
            for (const auto& command : commands_) {
                if (mayBeSubcommand
                    && (command.name == argv[i]
                        || (command.shortName != 0
                            && command.shortName == argv[i][0]
                            && argv[i][1] == 0))) {
                    if (parsedArgs.chosenCmd_ != CMD_UNSPECIFIED) {
                        std::string err =
                                "Trying to choose another subcommand '"
                                + command.name
                                + "' when one is already chosen ("
                                + commandName(parsedArgs.chosenCmd_) + ")";
                        throw ParseException(err);
                    }
                    parsedArgs.chosenCmd_            = command.cmd;
                    parsedArgs.cmdVals_[command.cmd] = {};
                    isSubcommand                     = true; // double-break
                    break;
                }
            }
            if (isSubcommand) {
                continue;
            }
            // it's not a command, so it must be a positional argument
            if (parsedArgs.chosenCmd_ == CMD_UNSPECIFIED) {
                throw ParseException(
                        "Trying to pass a positional argument before specifying a subcommand!");
            }
            auto positionalIdx = cmdPositionalIndex[parsedArgs.chosenCmd_]++;
            const auto positionalSize =
                    cmdPositionals_.at(parsedArgs.chosenCmd_).size();
            if (positionalSize == 0) {
                throw ParseException(
                        "Trying to pass a positional argument when no positional arguments are expected!");
            }
            if (positionalIdx >= positionalSize) {
                positionalIdx          = positionalSize - 1;
                const auto& positional = cmdPositionals_.at(
                        parsedArgs.chosenCmd_)[positionalIdx];
                if (positional.numArgs == NumArgs::ZeroOrOne
                    || positional.numArgs == NumArgs::One) {
                    std::ostringstream msg;
                    msg << "Too many positional arguments for command '"
                        << commandName(parsedArgs.chosenCmd_) << "'.\n";

                    // Show expected positional arguments
                    msg << "Expected " << positionalSize
                        << " positional argument"
                        << (positionalSize == 1 ? "" : "s") << ":";
                    for (size_t j = 0; j < positionalSize; ++j) {
                        msg << " <"
                            << cmdPositionals_.at(parsedArgs.chosenCmd_)[j].name
                            << ">";
                    }
                    msg << "\n";

                    // Show the problematic argument
                    msg << "Got unexpected positional argument '" << argv[i]
                        << "' at position " << (positionalIdx + 1);

                    throw ParseException({ msg.str() });
                }
            }
            auto positionalName =
                    cmdPositionals_.at(parsedArgs.chosenCmd_)[positionalIdx]
                            .name;
            parsedArgs.cmdVals_[parsedArgs.chosenCmd_][positionalName]
                    .push_back(argv[i]);
        }
    }
    return parsedArgs;
}

void ArgParser::validate(const ParsedArgs& parsedArgs) const
{
    if (parsedArgs.chosenCmd_ == CMD_UNSPECIFIED) {
        throw ParseException("No subcommand specified!");
    }
    for (const auto& positional : cmdPositionals_.at(parsedArgs.chosenCmd_)) {
        const auto& positionals = parsedArgs.cmdVals_.at(parsedArgs.chosenCmd_);

        // Validate required positional arguments are present
        if (positionals.count(positional.name) == 0
            || positionals.at(positional.name).size() == 0) {
            if (positional.numArgs == NumArgs::One
                || positional.numArgs == NumArgs::OneOrMore) {
                throw ParseException(
                        "Missing positional argument: " + positional.name);
            } else {
                continue;
            }
        }

        auto numArgs = positionals.at(positional.name).size();
        if (numArgs > 1 && positional.numArgs == NumArgs::One) {
            throw ParseException(
                    "Too many positional arguments for " + positional.name);
        }
    }
}

} // namespace openzl::arg
