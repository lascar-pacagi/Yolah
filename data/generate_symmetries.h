#ifndef GENERATE_SYMMETRIES_H
#define GENERATE_SYMMETRIES_H
#include <iostream>
#include <fstream>
#include <filesystem>

namespace data {
    void generate_symmetries(std::istream& is, std::ostream& os);
    void generate_symmetries(const std::string& src_dir, const std::string& dst_dir);
    void cut(const std::string& src_dir, const std::string& dst_dir, size_t nb_lines_per_file = 9000);
    void generate_symmetries(const std::filesystem::path& input_file, const std::filesystem::path& output_file);
    void generate_symmetries2(const std::string& src_dir, const std::string& dst_dir);
}

#endif
