// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "tools/sddl2/compiler/grouper/Grouper.h"

#include "tools/sddl2/compiler/Exception.h"
#include "tools/sddl2/compiler/Utils.h"

using namespace openzl::sddl2::detail;

namespace openzl::sddl2 {

namespace {
class GrouperImpl {
   public:
    explicit GrouperImpl(const Logger& logger) : log_(logger) {}

   private:
    GroupingPtr group_list_inner(
            GroupingVec::const_iterator& it,
            const GroupingVec::const_iterator end,
            const ListSymSet& list_sym_set) const
    {
        const auto& open_node = *it;
        if (*open_node != list_sym_set.open) {
            throw InvariantViolation(
                    open_node->loc(),
                    "List type '"
                            + std::string{ sym_to_repr_str(list_sym_set.open) }
                            + "..."
                            + std::string{ sym_to_repr_str(list_sym_set.sep) }
                            + " ..."
                            + std::string{ sym_to_repr_str(list_sym_set.close) }
                            + "' doesn't start with expected opening token!");
        }

        GroupingVec groups;
        GroupingVec cur_group_nodes;

        const auto group_cur = [&]() {
            cur_group_nodes.push_back(*it);
            groups.push_back(group_expr(std::move(cur_group_nodes)));
            cur_group_nodes.clear();
        };

        while (true) {
            ++it;
            if (it == end) {
                throw SyntaxError(
                        open_node->loc(),
                        "Couldn't find matching closing token '"
                                + std::string{ sym_to_repr_str(
                                        list_sym_set.close) }
                                + "' to close this list.");
            }

            if (**it == list_sym_set.close) {
                break;
            }

            // If we see a new list, group it
            auto maybe_list = maybe_group_list(it, end);
            if (maybe_list) {
                cur_group_nodes.push_back(std::move(maybe_list));
                continue;
            }

            // Finalize the current group if we see a separator
            if (**it == list_sym_set.sep) {
                if (cur_group_nodes.empty()) {
                    throw SyntaxError(
                            (*it)->loc(),
                            "Can't have an empty expression in the middle of a list.");
                }
                group_cur();
                continue;
            }

            if (**it == Symbol::NL) {
                // We don't use newlines as separators inside lists.
                continue;
            }

            cur_group_nodes.push_back(*it);
        }

        if (!cur_group_nodes.empty()) {
            group_cur();
        }

        return std::make_shared<GroupingList>(
                list_sym_set.type, open_node, *it, std::move(groups));
    }

    /**
     * Mutates @p it if a list is found, to point to the node that closes the
     * list.
     *
     * @returns a node representing the grouped list if found, a nullptr
     *          otherwise.
     */
    GroupingPtr maybe_group_list(
            GroupingVec::const_iterator& it,
            const GroupingVec::const_iterator end) const
    {
        const auto token = (*it)->as_token();
        if (token == nullptr) {
            return {};
        }
        if (!(**token).is_sym()) {
            return {};
        }
        auto lss_it = list_sym_sets.find((**token).sym());
        if (lss_it == list_sym_sets.end()) {
            return {};
        }
        return group_list_inner(it, end, lss_it->second);
    }

    // @p nodes includes statement terminator
    GroupingPtr group_expr(GroupingVec nodes) const
    {
        const auto full_loc = join_locs(nodes);
        const auto begin    = nodes.cbegin();
        const auto end      = nodes.cend();

        if (nodes.empty()) {
            throw InvariantViolation(
                    full_loc,
                    "Expression is empty, even though the token list should include at least the expression-ending token (e.g., ';', ',', ...).");
        }

        GroupingVec grouped;
        for (auto it = begin; it != end; ++it) {
            auto maybe_list = maybe_group_list(it, end);
            if (maybe_list) {
                grouped.push_back(std::move(maybe_list));
                continue;
            }
            grouped.push_back(*it);
        }

        auto terminator = nodes.back();
        nodes.pop_back();

        return std::make_shared<GroupingExpr>(
                std::move(nodes), std::move(terminator));
    }

    GroupingVec group_stmts(
            const GroupingVec::const_iterator begin,
            const GroupingVec::const_iterator end) const
    {
        GroupingVec stmts;
        GroupingVec nodes_in_cur_stmt;

        for (auto it = begin; it != end; ++it) {
            const auto* const token = (*it)->as_token();
            if (token != nullptr) {
                if (*token == Symbol::NL) {
                    if (!nodes_in_cur_stmt.empty()) {
                        nodes_in_cur_stmt.push_back(*it);
                        stmts.push_back(
                                group_expr(std::move(nodes_in_cur_stmt)));
                        nodes_in_cur_stmt.clear();
                        continue;
                    }
                    continue;
                }

                auto maybe_list = maybe_group_list(it, end);
                if (maybe_list) {
                    nodes_in_cur_stmt.push_back(std::move(maybe_list));
                    continue;
                }
            }

            // Otherwise
            nodes_in_cur_stmt.push_back(*it);
        }

        if (!nodes_in_cur_stmt.empty()) {
            // Implicitly terminate last expression, even if the explicit
            // terminator ';' or even the implicit terminator '\n' is missing.
            // I.e., don't require programs end in a new-line.
            nodes_in_cur_stmt.push_back(
                    std::make_shared<GroupingToken>(
                            Token{ SourceLocation::null(), Symbol::NL }));
            stmts.push_back(group_expr(std::move(nodes_in_cur_stmt)));
            nodes_in_cur_stmt.clear();
        }

        return stmts;
    }

   public:
    GroupingVec group(const std::vector<Token>& tokens) const
    {
        GroupingVec nodes;
        for (const auto& tok : tokens) {
            nodes.push_back(std::make_shared<GroupingToken>(tok));
        }
        // Top level of the source is a series of statements.
        nodes = group_stmts(nodes.begin(), nodes.end());

        log_(1) << "Groups:" << std::endl;
        for (const auto& node : nodes) {
            node->print(log_(1), 2);
        }
        log_(1) << std::endl;

        return nodes;
    }

   private:
    const Logger& log_;
};
} // anonymous namespace

Grouper::Grouper(const Logger& logger) : log_(logger) {}

GroupingVec Grouper::group(const std::vector<Token>& tokens) const
{
    return GrouperImpl{ log_ }.group(tokens);
}
} // namespace openzl::sddl2
