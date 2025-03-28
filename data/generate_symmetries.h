#ifndef GENERATE_SYMMETRIES_H
#define GENERATE_SYMMETRIES_H
#include <iostream>

namespace data {
    void generate_symmetries(std::istream& is, std::ostream& os);
    void generate_symmetries(const std::string& src_dir, const std::string& dst_dir);
    void cut(const std::string& src_dir, const std::string& dst_dir, size_t nb_lines_per_file = 9000);
}

#endif
