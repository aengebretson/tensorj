#include "ast_nodes.hpp"
#include <iostream> // For print methods
#include <iomanip>  // For std::setw

namespace JInterpreter {

// Helper for indenting
void print_indent(std::ostream& os, int indent) {
    for (int i = 0; i < indent; ++i) {
        os << "  ";
    }
}

void NounLiteralNode::print(std::ostream& os, int indent) const {
    print_indent(os, indent);
    os << "NounLiteralNode (" << location << "): ";
    std::visit([&os](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::nullptr_t>) {
            os << "nullptr";
        } else if constexpr (std::is_same_v<T, std::string>) {
            os << "'" << arg << "'";
        } else {
            os << arg;
        }
    }, value);
    os << std::endl;
}

void NameNode::print(std::ostream& os, int indent) const {
    print_indent(os, indent);
    os << "NameNode (" << location << "): " << name << std::endl;
}

void VerbNode::print(std::ostream& os, int indent) const {
    print_indent(os, indent);
    os << "VerbNode (" << location << "): " << identifier << std::endl;
}

void AdverbNode::print(std::ostream& os, int indent) const {
    print_indent(os, indent);
    os << "AdverbNode (" << location << "): " << identifier << std::endl;
}

void MonadicApplicationNode::print(std::ostream& os, int indent) const {
    print_indent(os, indent);
    os << "MonadicApplicationNode (" << location << "):" << std::endl;
    if (verb) verb->print(os, indent + 1); else { print_indent(os, indent+1); os << "<null verb>" << std::endl;}
    if (argument) argument->print(os, indent + 1); else { print_indent(os, indent+1); os << "<null argument>" << std::endl;}
}

void DyadicApplicationNode::print(std::ostream& os, int indent) const {
    print_indent(os, indent);
    os << "DyadicApplicationNode (" << location << "):" << std::endl;
    if (left_argument) left_argument->print(os, indent + 1); else { print_indent(os, indent+1); os << "<null left_argument>" << std::endl;}
    if (verb) verb->print(os, indent + 1); else { print_indent(os, indent+1); os << "<null verb>" << std::endl;}
    if (right_argument) right_argument->print(os, indent + 1); else { print_indent(os, indent+1); os << "<null right_argument>" << std::endl;}
}

void AdverbApplicationNode::print(std::ostream& os, int indent) const {
    print_indent(os, indent);
    os << "AdverbApplicationNode (" << location << "):" << std::endl;
    if (verb) verb->print(os, indent + 1); else { print_indent(os, indent+1); os << "<null verb>" << std::endl;}
    if (adverb) adverb->print(os, indent + 1); else { print_indent(os, indent+1); os << "<null adverb>" << std::endl;}
}

// Implement print methods for other AST nodes as you define them

} // namespace JInterpreter
