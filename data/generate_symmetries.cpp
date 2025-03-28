#include "generate_symmetries.h"
#include <regex>
#include <map>
#include <filesystem>
#include <fstream>

namespace data {
    void generate_symmetries(std::istream& is, std::ostream& os) {
        using namespace std;
        map<string, string> diag1{{"h1","h1"},{"h2","g1"},{"g1","h2"},{"h3","f1"},{"f1","h3"},{"h4","e1"},{"e1","h4"},{"h5","d1"},{"d1","h5"},{"h6","c1"},{"c1","h6"},{"h7","b1"},{"b1","h7"},{"h8","a1"},{"a1","h8"},{"g2","g2"},{"g3","f2"},{"f2","g3"},{"g4","e2"},{"e2","g4"},{"g5","d2"},{"d2","g5"},{"g6","c2"},{"c2","g6"},{"g7","b2"},{"b2","g7"},{"g8","a2"},{"a2","g8"},{"f3","f3"},{"f4","e3"},{"e3","f4"},{"f5","d3"},{"d3","f5"},{"f6","c3"},{"c3","f6"},{"f7","b3"},{"b3","f7"},{"f8","a3"},{"a3","f8"},{"e4","e4"},{"e5","d4"},{"d4","e5"},{"e6","c4"},{"c4","e6"},{"e7","b4"},{"b4","e7"},{"e8","a4"},{"a4","e8"},{"d5","d5"},{"d6","c5"},{"c5","d6"},{"d7","b5"},{"b5","d7"},{"d8","a5"},{"a5","d8"},{"c6","c6"},{"c7","b6"},{"b6","c7"},{"c8","a6"},{"a6","c8"},{"b7","b7"},{"b8","a7"},{"a7","b8"},{"a8","a8"},};
        map<string, string> diag2{{"a1","a1"},{"a2","b1"},{"b1","a2"},{"a3","c1"},{"c1","a3"},{"a4","d1"},{"d1","a4"},{"a5","e1"},{"e1","a5"},{"a6","f1"},{"f1","a6"},{"a7","g1"},{"g1","a7"},{"a8","h1"},{"h1","a8"},{"b2","b2"},{"b3","c2"},{"c2","b3"},{"b4","d2"},{"d2","b4"},{"b5","e2"},{"e2","b5"},{"b6","f2"},{"f2","b6"},{"b7","g2"},{"g2","b7"},{"b8","h2"},{"h2","b8"},{"c3","c3"},{"c4","d3"},{"d3","c4"},{"c5","e3"},{"e3","c5"},{"c6","f3"},{"f3","c6"},{"c7","g3"},{"g3","c7"},{"c8","h3"},{"h3","c8"},{"d4","d4"},{"d5","e4"},{"e4","d5"},{"d6","f4"},{"f4","d6"},{"d7","g4"},{"g4","d7"},{"d8","h4"},{"h4","d8"},{"e5","e5"},{"e6","f5"},{"f5","e6"},{"e7","g5"},{"g5","e7"},{"e8","h5"},{"h5","e8"},{"f6","f6"},{"f7","g6"},{"g6","f7"},{"f8","h6"},{"h6","f8"},{"g7","g7"},{"g8","h7"},{"h7","g8"},{"h8","h8"},};
        map<string, string> central{{"d4","e5"},{"f8","c1"},{"h3","a6"},{"d1","e8"},{"b2","g7"},{"f7","c2"},{"c3","f6"},{"b5","g4"},{"a3","h6"},{"h8","a1"},{"g8","b1"},{"h1","a8"},{"h7","a2"},{"g2","b7"},{"g5","b4"},{"h2","a7"},{"e6","d3"},{"c8","f1"},{"c1","f8"},{"b1","g8"},{"b3","g6"},{"d7","e2"},{"g1","b8"},{"b4","g5"},{"c5","f4"},{"e1","d8"},{"f1","c8"},{"b7","g2"},{"c4","f5"},{"a6","h3"},{"d5","e4"},{"e5","d4"},{"h5","a4"},{"d8","e1"},{"h6","a3"},{"b8","g1"},{"e3","d6"},{"f5","c4"},{"f6","c3"},{"d3","e6"},{"a2","h7"},{"a8","h1"},{"a1","h8"},{"a4","h5"},{"c6","f3"},{"c2","f7"},{"f2","c7"},{"h4","a5"},{"g6","b3"},{"a5","h4"},{"g3","b6"},{"e4","d5"},{"e7","d2"},{"c7","f2"},{"g4","b5"},{"f3","c6"},{"g7","b2"},{"a7","h2"},{"d2","e7"},{"e2","d7"},{"b6","g3"},{"e8","d1"},{"d6","e3"},{"f4","c5"},};
        regex re_moves(R"(((\w\d):(\w\d))+)", regex_constants::ECMAScript);
        regex re_scores(R"((\d+)/(\d+))", regex_constants::ECMAScript);
        while (is) {
            string line;
            getline(is, line);
            if (line == "") continue;
            for (auto it = sregex_iterator(begin(line), end(line), re_moves); it != sregex_iterator(); ++it) {
                smatch match = *it;
                string match_str = match.str();
                string sq1 = match[2].str();
                string sq2 = match[3].str();
                if (sq1 == "a1" && sq2 == "a1") {
                    os << "a1:a1" << ' ';
                } else {
                    os << diag1[sq1] << ':' << diag1[sq2] << ' ';
                }                
            }
            smatch match;
            regex_search(line, match, re_scores);
            size_t black_score = atoi(match[1].str().c_str());
            size_t white_score = atoi(match[2].str().c_str());
            os << "(" << black_score << '/' << white_score << ")\n";
            for (auto it = sregex_iterator(begin(line), end(line), re_moves); it != sregex_iterator(); ++it) {
                smatch match = *it;
                string match_str = match.str();
                os << diag2[match[2].str()] << ':' << diag2[match[3].str()] << ' ';
            }
            os << "(" << black_score << '/' << white_score << ")\n";
            for (auto it = sregex_iterator(begin(line), end(line), re_moves); it != sregex_iterator(); ++it) {
                smatch match = *it;
                string match_str = match.str();
                string sq1 = match[2].str();
                string sq2 = match[3].str();
                if (sq1 == "a1" && sq2 == "a1") {
                    os << "a1:a1" << ' ';
                } else {
                    os << central[sq1] << ':' << central[sq2] << ' ';
                }                
            }
            os << "(" << black_score << '/' << white_score << ")\n";
        }
    }

    void cut(const std::string& src_dir, const std::string& dst_dir, size_t nb_lines_per_file) {
        const std::filesystem::path src(src_dir);
        const std::filesystem::path dst(dst_dir);
        std::regex re_games("^games((?!.*symmetries.*))", std::regex_constants::ECMAScript|std::regex_constants::multiline);
        for (auto const& dir_entry : std::filesystem::directory_iterator(src_dir)) {
            auto path = dir_entry.path();
            if (std::filesystem::is_directory(path) || !std::regex_search(path.filename().string(), re_games)) continue;      
            std::cout << path << std::endl;
            auto input = std::ifstream(path);
            size_t count = 0;
            while (input) {                
                auto output = std::ofstream(dst_dir / path.filename().replace_extension(std::to_string(count++) + ".txt"));
                size_t i = 0;
                while (input && i < nb_lines_per_file) {
                    std::string line;
                    std::getline(input, line);
                    if (line == "") continue;
                    output << line << '\n';
                    i++;
                }
            }            
        }
    }

    void generate_symmetries(const std::string& src_dir, const std::string& dst_dir) {
        const std::filesystem::path src(src_dir);
        const std::filesystem::path dst(dst_dir);
        std::regex re_games("^games((?!.*symmetries.*))", std::regex_constants::ECMAScript|std::regex_constants::multiline);
        for (auto const& dir_entry : std::filesystem::directory_iterator(src_dir)) {
            auto path = dir_entry.path();
            if (!std::regex_search(path.filename().string(), re_games)) continue;      
            std::cout << path << std::endl;
            auto input = std::ifstream(path);
            auto output = std::ofstream(dst_dir / path.filename().replace_extension("symmetries.txt"));
            generate_symmetries(input, output);
        }
    }
}
