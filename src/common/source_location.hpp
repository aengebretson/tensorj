#ifndef J_INTERPRETER_SOURCE_LOCATION_HPP
#define J_INTERPRETER_SOURCE_LOCATION_HPP

#include <string>
#include <ostream>

namespace JInterpreter {

struct SourceLocation {
    int line = 1;
    int column = 1;
    std::string file_name; // Optional: for multi-file projects

    SourceLocation() = default;
    SourceLocation(int l, int c, std::string fn = "")
        : line(l), column(c), file_name(std::move(fn)) {}

    friend std::ostream& operator<<(std::ostream& os, const SourceLocation& loc) {
        os << (loc.file_name.empty() ? "" : loc.file_name + ":")
           << loc.line << ":" << loc.column;
        return os;
    }
};

} // namespace JInterpreter

#endif // J_INTERPRETER_SOURCE_LOCATION_HPP
