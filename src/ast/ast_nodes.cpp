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

void ConjunctionNode::print(std::ostream& os, int indent) const {
    print_indent(os, indent);
    os << "ConjunctionNode (" << location << "): " << identifier << std::endl;
}

void ConjunctionApplicationNode::print(std::ostream& os, int indent) const {
    print_indent(os, indent);
    os << "ConjunctionApplicationNode (" << location << "):" << std::endl;
    if (left_operand) left_operand->print(os, indent + 1); else { print_indent(os, indent+1); os << "<null left_operand>" << std::endl;}
    if (conjunction) conjunction->print(os, indent + 1); else { print_indent(os, indent+1); os << "<null conjunction>" << std::endl;}
    if (right_operand) right_operand->print(os, indent + 1); else { print_indent(os, indent+1); os << "<null right_operand>" << std::endl;}
}

void VectorLiteralNode::print(std::ostream& os, int indent) const {
    print_indent(os, indent);
    os << "VectorLiteralNode (" << location << "): [";
    for (size_t i = 0; i < elements.size(); ++i) {
        if (i > 0) os << " ";
        std::visit([&os](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, std::nullptr_t>) {
                os << "nullptr";
            } else if constexpr (std::is_same_v<T, std::string>) {
                os << "'" << arg << "'";
            } else {
                os << arg;
            }
        }, elements[i]);
    }
    os << "]" << std::endl;
}

// Implement print methods for other AST nodes as you define them

} // namespace JInterpreter
