#include "analyze_games.h"
#include <string>
#include <regex>

namespace data {
    void analyze_games(std::istream& is, std::ostream& os) {
        using namespace std;
        regex re_moves(R"(((\w\d):(\w\d))+)", regex_constants::ECMAScript);
        regex re_scores(R"((\d+)/(\d+))", regex_constants::ECMAScript);
        size_t black_victories = 0;
        size_t white_victories = 0;
        size_t draws = 0;
        size_t max_black_score = 0;
        size_t max_white_score = 0;
        while (is) {
            string line;
            getline(is, line);
            // for (auto it = sregex_iterator(begin(line), end(line), re_moves); it != sregex_iterator(); ++it) {
            //     smatch match = *it;
            //     string match_str = match.str();
            //     os << match_str << '\n';
            // }
            smatch match;
            regex_search(line, match, re_scores);
            if (match.size() == 3) {
                size_t black_score = atoi(match[1].str().c_str());
                size_t white_score = atoi(match[2].str().c_str());
                black_victories += black_score > white_score;
                white_victories += white_score > black_score;
                draws += black_score == white_score;
                max_black_score = max(max_black_score, black_score);
                max_white_score = max(max_white_score, white_score);
            }                        
        }
        double n = black_victories + white_victories + draws;
        os << "[nb games       : " << n << "]\n";
        os << "[black victories: " << (black_victories / n) << "]\n";
        os << "[white victories: " << (white_victories / n) << "]\n";
        os << "[draws          : " << (draws / n) << "]\n";
        os << "[black max score: " << max_black_score << "]\n";
        os << "[white max score: " << max_white_score << "]\n";
    }
}