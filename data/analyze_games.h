#ifndef ANALYZE_GAMES_H
#define ANALYZE_GAMES_H
#include <iostream>

namespace data {
    void analyze_games(std::istream& is, std::ostream& os);
    void analyze_games2(const std::string& src_dir, std::ostream& os);
}

#endif