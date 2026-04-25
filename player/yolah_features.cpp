#include "yolah_features.h"
#include "game.h"
#include "generate_games.h"
#include "types.h"
#include <bit>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <regex>

namespace YolahFeatures {
uint64_t floodfill(uint64_t player_bb, uint64_t free_bb) {
  uint64_t prev_flood = 0;
  uint64_t flood = player_bb;
  while (prev_flood != flood) {
    prev_flood = flood;
    flood |= shift_all_directions(flood) & free_bb;
  }
  flood ^= player_bb;
  return flood;
}

uint8_t alone(uint64_t player_bb, uint64_t flood_opponent,
              const uint64_t flood_pieces[4]) {
  uint64_t total_player_flood = 0;
  uint8_t res = 0;
  for (int i = 0; i < 4; i++) {
    uint64_t flood = flood_pieces[i];
    res += ((flood & flood_opponent) == 0) *
           std::popcount(flood & ~total_player_flood);
    total_player_flood |= flood;
  }
  return res;
}

std::pair<uint64_t, uint64_t> influence(uint64_t black_bb, uint64_t white_bb,
                                        uint64_t free_bb) {
  auto one_step = [&](uint64_t flood, uint64_t free) {
    return shift_all_directions(flood) & free;
  };
  uint64_t prev_black_influence = 0;
  uint64_t prev_white_influence = 0;
  uint64_t black_influence = black_bb;
  uint64_t white_influence = white_bb;
  uint64_t black_frontier = black_influence;
  uint64_t white_frontier = white_influence;
  uint64_t neutral = 0;
  while ((prev_black_influence != black_influence) ||
         (prev_white_influence != white_influence)) {
    black_frontier = one_step(black_frontier, free_bb) & ~white_influence;
    white_frontier = one_step(white_frontier, free_bb) & ~black_influence;
    neutral |= one_step(neutral, free_bb) | (black_frontier & white_frontier);
    black_frontier &= ~neutral;
    white_frontier &= ~neutral;
    prev_black_influence = black_influence;
    prev_white_influence = white_influence;
    black_influence |= black_frontier;
    white_influence |= white_frontier;
  }
  return {black_influence, white_influence};
}

uint8_t count_groups(uint64_t player_bb, const uint64_t pieces_bb[4],
                     const uint64_t flood_pieces[4]) {
  int groups = 0;
  for (int i = 0; i < 4; i++) {
    groups += (player_bb & pieces_bb[i]) != 0;
    player_bb &= ~(flood_pieces[i] | shift_all_directions(flood_pieces[i]) |
                   pieces_bb[i]);
  }
  return groups;
}

void set_features(uint8_t* features,
                  const Yolah &yolah) {
#define NDEBUG
  using namespace std;
  const auto [black_moves_bb0, black_moves_bb1, black_moves_bb2,
              black_moves_bb3] = yolah.moves_bb(Yolah::BLACK);
  const auto [white_moves_bb0, white_moves_bb1, white_moves_bb2,
              white_moves_bb3] = yolah.moves_bb(Yolah::WHITE);
  uint64_t black_moves_bb =
      black_moves_bb0 | black_moves_bb1 | black_moves_bb2 | black_moves_bb3;
  uint64_t white_moves_bb =
      white_moves_bb0 | white_moves_bb1 | white_moves_bb2 | white_moves_bb3;
  int black_nb_moves = popcount(black_moves_bb0) + popcount(black_moves_bb1) +
                       popcount(black_moves_bb2) + popcount(black_moves_bb3);
  int white_nb_moves = popcount(white_moves_bb0) + popcount(white_moves_bb1) +
                       popcount(white_moves_bb2) + popcount(white_moves_bb3);
  features[NO_MOVE_BLACK] = black_nb_moves == 0;
  features[NO_MOVE_WHITE] = white_nb_moves == 0;
#ifndef NDEBUG
  string _;
  cerr << "No move\n";
  cerr << yolah << '\n';
  cerr << format("NO_MOVE_BLACK: {}\n", features[NO_MOVE_BLACK]);
  cerr << format("NO_MOVE_WHITE: {}\n", features[NO_MOVE_WHITE]);
  getline(cin, _);
#endif
  features[MOVE_BLACK] = black_nb_moves;
  features[MOVE_WHITE] = white_nb_moves;
#ifndef NDEBUG
  cerr << "Move\n";
  cerr << yolah << '\n';
  cerr << format("MOVE_BLACK: {}\n", features[MOVE_BLACK]);
  cerr << format("MOVE_WHITE: {}\n", features[MOVE_WHITE]);
  getline(cin, _);
#endif
  uint64_t black_bb = yolah.bitboard(Yolah::BLACK);
  uint64_t white_bb = yolah.bitboard(Yolah::WHITE);
  uint64_t free_bb = yolah.free_squares();
  uint64_t flood_black_pieces[4];
  uint64_t flood_white_pieces[4];
  uint64_t black_pieces_bb[4];
  uint64_t white_pieces_bb[4];
  {
    uint64_t player_bb = black_bb;
    uint64_t bb = player_bb & -player_bb;
    black_pieces_bb[0] = bb;
    flood_black_pieces[0] = floodfill(bb, free_bb);
    player_bb &= ~bb;
    bb = player_bb & -player_bb;
    black_pieces_bb[1] = bb;
    flood_black_pieces[1] = floodfill(bb, free_bb);
    player_bb &= ~bb;
    bb = player_bb & -player_bb;
    black_pieces_bb[2] = bb;
    flood_black_pieces[2] = floodfill(bb, free_bb);
    player_bb &= ~bb;
    bb = player_bb & -player_bb;
    black_pieces_bb[3] = bb;
    flood_black_pieces[3] = floodfill(bb, free_bb);
  }
  {
    uint64_t player_bb = white_bb;
    uint64_t bb = player_bb & -player_bb;
    white_pieces_bb[0] = bb;
    flood_white_pieces[0] = floodfill(bb, free_bb);
    player_bb &= ~bb;
    bb = player_bb & -player_bb;
    white_pieces_bb[1] = bb;
    flood_white_pieces[1] = floodfill(bb, free_bb);
    player_bb &= ~bb;
    bb = player_bb & -player_bb;
    white_pieces_bb[2] = bb;
    flood_white_pieces[2] = floodfill(bb, free_bb);
    player_bb &= ~bb;
    bb = player_bb & -player_bb;
    white_pieces_bb[3] = bb;
    flood_white_pieces[3] = floodfill(bb, free_bb);
  }
  uint64_t flood_black = flood_black_pieces[0] | flood_black_pieces[1] |
                         flood_black_pieces[2] | flood_black_pieces[3];
  uint64_t flood_white = flood_white_pieces[0] | flood_white_pieces[1] |
                         flood_white_pieces[2] | flood_white_pieces[3];
  features[CONNECTIVITY_BLACK] =
      popcount(flood_black_pieces[0]) + popcount(flood_black_pieces[1]) +
      popcount(flood_black_pieces[2]) + popcount(flood_black_pieces[3]);
  features[CONNECTIVITY_WHITE] =
      popcount(flood_white_pieces[0]) + popcount(flood_white_pieces[1]) +
      popcount(flood_white_pieces[2]) + popcount(flood_white_pieces[3]);
#ifndef NDEBUG
  cerr << "Connectivity\n";
  cerr << yolah << '\n';
  cerr << "BLACK\n";
  for (int i = 0; i < 4; i++) {
    cerr << Bitboard::pretty(flood_black_pieces[i]) << '\n';
  }
  cerr << format("CONNECTIVITY_BLACK: {}\n", features[CONNECTIVITY_BLACK]);
  getline(cin, _);
  cerr << "WHITE\n";
  for (int i = 0; i < 4; i++) {
    cerr << Bitboard::pretty(flood_white_pieces[i]) << '\n';
  }
  cerr << format("CONNECTIVITY_WHITE: {}\n", features[CONNECTIVITY_WHITE]);
  getline(cin, _);
#endif
  features[CONNECTIVITY_SET_BLACK] = popcount(flood_black);
  features[CONNECTIVITY_SET_WHITE] = popcount(flood_white);
#ifndef NDEBUG
  cerr << "Connectivity set\n";
  cerr << yolah << '\n';
  cerr << "BLACK\n";
  cerr << Bitboard::pretty(flood_black) << '\n';
  cerr << format("CONNECTIVITY_SET_BLACK: {}\n",
                 features[CONNECTIVITY_SET_BLACK]);
  getline(cin, _);
  cerr << "WHITE\n";
  cerr << Bitboard::pretty(flood_white) << '\n';
  cerr << format("CONNECTIVITY_SET_WHITE: {}\n",
                 features[CONNECTIVITY_SET_WHITE]);
  getline(cin, _);
#endif
  features[ALONE_BLACK] = alone(black_bb, flood_white, flood_black_pieces);
  features[ALONE_WHITE] = alone(white_bb, flood_black, flood_white_pieces);
#ifndef NDEBUG
  cerr << "Alone\n";
  cerr << yolah << '\n';
  cerr << format("ALONE_BLACK: {}\n", features[ALONE_BLACK]);
  cerr << format("ALONE_WHITE: {}\n", features[ALONE_WHITE]);
  getline(cin, _);
#endif
  features[FIRST_BLACK] = popcount(black_moves_bb & ~white_moves_bb);
  features[FIRST_WHITE] = popcount(white_moves_bb & ~black_moves_bb);
#ifndef NDEBUG
  cerr << "First\n";
  cerr << yolah << '\n';
  cerr << "BLACK\n";
  cerr << Bitboard::pretty(black_moves_bb & ~white_moves_bb) << '\n';
  cerr << format("FIRST_BLACK: {}\n", features[FIRST_BLACK]);
  getline(cin, _);
  cerr << "WHITE\n";
  cerr << Bitboard::pretty(white_moves_bb & ~black_moves_bb) << '\n';
  cerr << format("FIRST_WHITE: {}\n", features[FIRST_WHITE]);
  getline(cin, _);
#endif
  const auto [black_influence, white_influence] =
      influence(black_bb, white_bb, free_bb);
  features[INFLUENCE_BLACK] = popcount(black_influence);
  features[INFLUENCE_WHITE] = popcount(white_influence);
#ifndef NDEBUG
  cerr << "Influence\n";
  cerr << yolah << '\n';
  cerr << "BLACK\n";
  cerr << Bitboard::pretty(black_influence) << '\n';
  cerr << format("INFLUENCE_BLACK: {}\n", features[INFLUENCE_BLACK]);
  getline(cin, _);
  cerr << "WHITE\n";
  cerr << Bitboard::pretty(white_influence) << '\n';
  cerr << format("INFLUENCE_WHITE: {}\n", features[INFLUENCE_WHITE]);
  getline(cin, _);
#endif
  features[BLOCKED_BLACK] = (black_moves_bb0 == 0) + (black_moves_bb1 == 0) +
                            (black_moves_bb2 == 0) + (black_moves_bb3 == 0);
  features[BLOCKED_WHITE] = (white_moves_bb0 == 0) + (white_moves_bb1 == 0) +
                            (white_moves_bb2 == 0) + (white_moves_bb3 == 0);
#ifndef NDEBUG
  cerr << "Blocked\n";
  cerr << yolah << '\n';
  cerr << format("BLOCKED_BLACK: {}\n", features[BLOCKED_BLACK]);
  cerr << format("BLOCKED_WHITE: {}\n", features[BLOCKED_WHITE]);
  getline(cin, _);
#endif
  int black_piece0_freedom = popcount(shift_all_directions(black_pieces_bb[0]));
  int black_piece1_freedom = popcount(shift_all_directions(black_pieces_bb[1]));
  int black_piece2_freedom = popcount(shift_all_directions(black_pieces_bb[2]));
  int black_piece3_freedom = popcount(shift_all_directions(black_pieces_bb[3]));
  features[FREEDOM_LOW_BLACK] =
      (black_piece0_freedom <= 2) + (black_piece1_freedom <= 2) +
      (black_piece2_freedom <= 2) + (black_piece3_freedom <= 2);
  features[FREEDOM_MID_BLACK] =
      (2 < black_piece0_freedom && black_piece0_freedom <= 5) +
      (2 < black_piece1_freedom && black_piece1_freedom <= 5) +
      (2 < black_piece2_freedom && black_piece2_freedom <= 5) +
      (2 < black_piece3_freedom && black_piece3_freedom <= 5);
  features[FREEDOM_HIGH_BLACK] =
      (5 < black_piece0_freedom) + (5 < black_piece1_freedom) +
      (5 < black_piece2_freedom) + (5 < black_piece3_freedom);
  int white_piece0_freedom = popcount(shift_all_directions(white_pieces_bb[0]));
  int white_piece1_freedom = popcount(shift_all_directions(white_pieces_bb[1]));
  int white_piece2_freedom = popcount(shift_all_directions(white_pieces_bb[2]));
  int white_piece3_freedom = popcount(shift_all_directions(white_pieces_bb[3]));
  features[FREEDOM_LOW_WHITE] =
      (white_piece0_freedom <= 2) + (white_piece1_freedom <= 2) +
      (white_piece2_freedom <= 2) + (white_piece3_freedom <= 2);
  features[FREEDOM_MID_WHITE] =
      (2 < white_piece0_freedom && white_piece0_freedom <= 5) +
      (2 < white_piece1_freedom && white_piece1_freedom <= 5) +
      (2 < white_piece2_freedom && white_piece2_freedom <= 5) +
      (2 < white_piece3_freedom && white_piece3_freedom <= 5);
  features[FREEDOM_HIGH_WHITE] =
      (5 < white_piece0_freedom) + (5 < white_piece1_freedom) +
      (5 < white_piece2_freedom) + (5 < white_piece3_freedom);
#ifndef NDEBUG
  cerr << "Freedom\n";
  cerr << yolah << '\n';
  cerr << "BLACK\n";
  cerr << format("FREEDOM_LOW_BLACK: {}\n", features[FREEDOM_LOW_BLACK]);
  cerr << format("FREEDOM_MID_BLACK: {}\n", features[FREEDOM_MID_BLACK]);
  cerr << format("FREEDOM_HIGH_BLACK: {}\n", features[FREEDOM_HIGH_BLACK]);
  getline(cin, _);
  cerr << "WHITE\n";
  cerr << format("FREEDOM_LOW_WHITE: {}\n", features[FREEDOM_LOW_WHITE]);
  cerr << format("FREEDOM_MID_WHITE: {}\n", features[FREEDOM_MID_WHITE]);
  cerr << format("FREEDOM_HIGH_WHITE: {}\n", features[FREEDOM_HIGH_WHITE]);
  getline(cin, _);
#endif
  features[GROUP_BLACK] =
      count_groups(black_bb, black_pieces_bb, flood_black_pieces);
  features[GROUP_WHITE] =
      count_groups(white_bb, white_pieces_bb, flood_white_pieces);
#ifndef NDEBUG
  cerr << "Group\n";
  cerr << yolah << '\n';
  cerr << format("GROUP_BLACK: {}\n", features[GROUP_BLACK]);
  cerr << format("GROUP_WHITE: {}\n", features[GROUP_WHITE]);
  getline(cin, _);
#endif
  static constexpr uint64_t CENTER = 0x00003C3C3C3C0000ULL;
  static constexpr uint64_t LOWER_LEFT_CORNER = 0x0000000000000303ULL;
  static constexpr uint64_t LOWER_RIGHT_CORNER = 0x000000000000C0C0ULL;
  static constexpr uint64_t UPPER_RIGHT_CORNER = 0xC0C0000000000000ULL;
  static constexpr uint64_t UPPER_LEFT_CORNER = 0x0303000000000000ULL;
  static constexpr uint64_t LOWER_MIDDLE = 0x0000000000003C3CULL;
  static constexpr uint64_t RIGHT_MIDDLE = 0x0000C0C0C0C00000ULL;
  static constexpr uint64_t UPPER_MIDDLE = 0x3C3C000000000000ULL;
  static constexpr uint64_t LEFT_MIDDLE = 0x0000030303030000ULL;
  features[CENTER_BLACK] = popcount(black_bb & CENTER);
  features[CENTER_WHITE] = popcount(white_bb & CENTER);
  features[LOWER_LEFT_CORNER_BLACK] = popcount(black_bb & LOWER_LEFT_CORNER);
  features[LOWER_LEFT_CORNER_WHITE] = popcount(white_bb & LOWER_LEFT_CORNER);
  features[LOWER_RIGHT_CORNER_BLACK] = popcount(black_bb & LOWER_RIGHT_CORNER);
  features[LOWER_RIGHT_CORNER_WHITE] = popcount(white_bb & LOWER_RIGHT_CORNER);
  features[UPPER_LEFT_CORNER_BLACK] = popcount(black_bb & UPPER_LEFT_CORNER);
  features[UPPER_LEFT_CORNER_WHITE] = popcount(white_bb & UPPER_LEFT_CORNER);
  features[UPPER_RIGHT_CORNER_BLACK] = popcount(black_bb & UPPER_RIGHT_CORNER);
  features[UPPER_RIGHT_CORNER_WHITE] = popcount(white_bb & UPPER_RIGHT_CORNER);
  features[LOWER_MIDDLE_BLACK] = popcount(black_bb & LOWER_MIDDLE);
  features[LOWER_MIDDLE_WHITE] = popcount(white_bb & LOWER_MIDDLE);
  features[RIGHT_MIDDLE_BLACK] = popcount(black_bb & RIGHT_MIDDLE);
  features[RIGHT_MIDDLE_WHITE] = popcount(white_bb & RIGHT_MIDDLE);
  features[UPPER_MIDDLE_BLACK] = popcount(black_bb & UPPER_MIDDLE);
  features[UPPER_MIDDLE_WHITE] = popcount(white_bb & UPPER_MIDDLE);
  features[LEFT_MIDDLE_BLACK] = popcount(black_bb & LEFT_MIDDLE);
  features[LEFT_MIDDLE_WHITE] = popcount(white_bb & LEFT_MIDDLE);
#ifndef NDEBUG
  cerr << "Position\n";
  cerr << yolah << '\n';
  cerr << "Center\n";
  cerr << Bitboard::pretty(CENTER) << '\n';
  cerr << format("CENTER_BLACK: {}\n", features[CENTER_BLACK]);
  cerr << format("CENTER_WHITE: {}\n", features[CENTER_WHITE]);
  getline(cin, _);
  cerr << yolah << '\n';
  cerr << "Lower left corner\n";
  cerr << Bitboard::pretty(LOWER_LEFT_CORNER) << '\n';
  cerr << format("LOWER_LEFT_CORNER_BLACK: {}\n",
                 features[LOWER_LEFT_CORNER_BLACK]);
  cerr << format("LOWER_LEFT_CORNER_WHITE: {}\n",
                 features[LOWER_LEFT_CORNER_WHITE]);
  getline(cin, _);
  cerr << yolah << '\n';
  cerr << "Lower right corner\n";
  cerr << Bitboard::pretty(LOWER_RIGHT_CORNER) << '\n';
  cerr << format("LOWER_RIGHT_CORNER_BLACK: {}\n",
                 features[LOWER_RIGHT_CORNER_BLACK]);
  cerr << format("LOWER_RIGHT_CORNER_WHITE: {}\n",
                 features[LOWER_RIGHT_CORNER_WHITE]);
  getline(cin, _);
  cerr << yolah << '\n';
  cerr << "Upper left corner\n";
  cerr << Bitboard::pretty(UPPER_LEFT_CORNER) << '\n';
  cerr << format("UPPER_LEFT_CORNER_BLACK: {}\n",
                 features[UPPER_LEFT_CORNER_BLACK]);
  cerr << format("UPPER_LEFT_CORNER_WHITE: {}\n",
                 features[UPPER_LEFT_CORNER_WHITE]);
  getline(cin, _);
  cerr << yolah << '\n';
  cerr << "Upper right corner\n";
  cerr << Bitboard::pretty(UPPER_RIGHT_CORNER) << '\n';
  cerr << format("UPPER_RIGHT_CORNER_BLACK: {}\n",
                 features[UPPER_RIGHT_CORNER_BLACK]);
  cerr << format("UPPER_RIGHT_CORNER_WHITE: {}\n",
                 features[UPPER_RIGHT_CORNER_WHITE]);
  getline(cin, _);
  cerr << yolah << '\n';
  cerr << "Lower middle\n";
  cerr << Bitboard::pretty(LOWER_MIDDLE) << '\n';
  cerr << format("LOWER_MIDDLE_BLACK: {}\n", features[LOWER_MIDDLE_BLACK]);
  cerr << format("LOWER_MIDDLE_WHITE: {}\n", features[LOWER_MIDDLE_WHITE]);
  getline(cin, _);
  cerr << yolah << '\n';
  cerr << "Right middle\n";
  cerr << Bitboard::pretty(RIGHT_MIDDLE) << '\n';
  cerr << format("RIGHT_MIDDLE_BLACK: {}\n", features[RIGHT_MIDDLE_BLACK]);
  cerr << format("RIGHT_MIDDLE_WHITE: {}\n", features[RIGHT_MIDDLE_WHITE]);
  getline(cin, _);
  cerr << yolah << '\n';
  cerr << "Upper middle\n";
  cerr << Bitboard::pretty(UPPER_MIDDLE) << '\n';
  cerr << format("UPPER_MIDDLE_BLACK: {}\n", features[UPPER_MIDDLE_BLACK]);
  cerr << format("UPPER_MIDDLE_WHITE: {}\n", features[UPPER_MIDDLE_WHITE]);
  getline(cin, _);
  cerr << yolah << '\n';
  cerr << "Left middle\n";
  cerr << Bitboard::pretty(LEFT_MIDDLE) << '\n';
  cerr << format("LEFT_MIDDLE_BLACK: {}\n", features[LEFT_MIDDLE_BLACK]);
  cerr << format("LEFT_MIDDLE_WHITE: {}\n", features[LEFT_MIDDLE_WHITE]);
  getline(cin, _);
#endif

  int stretch_black = 0;
  int teaming_black = 0;
  for (int i = 0; i < 4; i++) {
    for (int j = i + 1; j < 4; j++) {
      stretch_black +=
          !(flood_black_pieces[i] & flood_black_pieces[j]) *
          (popcount(flood_black_pieces[i] | flood_black_pieces[j]));
      teaming_black +=
          ((flood_black_pieces[i] & flood_black_pieces[j]) != 0) *
          (popcount(flood_black_pieces[i] | flood_black_pieces[j]));      
    }
  }
  features[STRETCH_BLACK] = stretch_black;
  features[TEAMING_BLACK] = teaming_black;
  
  int stretch_white = 0;
  int teaming_white = 0;
  for (int i = 0; i < 4; i++) {
    for (int j = i + 1; j < 4; j++) {
      stretch_white +=
          !(flood_white_pieces[i] & flood_white_pieces[j]) *
          (popcount(flood_white_pieces[i] | flood_white_pieces[j]));
      teaming_white +=
          ((flood_white_pieces[i] & flood_white_pieces[j]) != 0) *
          (popcount(flood_white_pieces[i] | flood_white_pieces[j]));      
    }
  }
  features[STRETCH_WHITE] = stretch_white;
  features[TEAMING_WHITE] = teaming_white;
  
#ifndef NDEBUG
  cerr << "Stretch\n";
  cerr << yolah << '\n';
  cerr << format("STRETCH_BLACK: {}\n", features[STRETCH_BLACK]);
  cerr << format("STRETCH_WHITE: {}\n", features[STRETCH_WHITE]);
  getline(cin, _);
#endif

#ifndef NDEBUG
  cerr << "Teaming\n";
  cerr << yolah << '\n';
  cerr << format("TEAMING_BLACK: {}\n", features[TEAMING_BLACK]);
  cerr << format("TEAMING_WHITE: {}\n", features[TEAMING_WHITE]);
  getline(cin, _);
#endif

  uint64_t black_bb_shift_all_dirs = shift_all_directions(black_bb);
  uint64_t white_bb_shift_all_dirs = shift_all_directions(white_bb);

  features[CONTACT_WITH_OWN_BLACK] =
      popcount(black_bb_shift_all_dirs & black_bb);
  features[CONTACT_WITH_OWN_WHITE] =
      popcount(white_bb_shift_all_dirs & white_bb);

  uint64_t hole_bb = yolah.empty_bitboard();

  features[CONTACT_WITH_HOLE_BLACK] =
      popcount(black_bb_shift_all_dirs & hole_bb);
  features[CONTACT_WITH_HOLE_WHITE] =
      popcount(white_bb_shift_all_dirs & hole_bb);

  features[CONTACT_WITH_FREE_BLACK] =
      popcount(black_bb_shift_all_dirs & free_bb);
  features[CONTACT_WITH_FREE_WHITE] =
      popcount(white_bb_shift_all_dirs & free_bb);  
  
  features[CONTACT_WITH_OTHER] = popcount(black_bb_shift_all_dirs & white_bb);

#ifndef NDEBUG
  cerr << "Contact\n";
  cerr << yolah << '\n';
  cerr << format("CONTACT_WITH_OWN_BLACK: {}\n",
                 features[CONTACT_WITH_OWN_BLACK]);
  cerr << format("CONTACT_WITH_OWN_WHITE: {}\n",
                 features[CONTACT_WITH_OWN_WHITE]);
  cerr << format("CONTACT_WITH_HOLE_BLACK: {}\n",
                 features[CONTACT_WITH_HOLE_BLACK]);
  cerr << format("CONTACT_WITH_HOLE_WHITE: {}\n",
                 features[CONTACT_WITH_HOLE_WHITE]);
  cerr << format("CONTACT_WITH_FREE_BLACK: {}\n",
                 features[CONTACT_WITH_FREE_BLACK]);
  cerr << format("CONTACT_WITH_FREE_WHITE: {}\n",
                 features[CONTACT_WITH_FREE_WHITE]);
  cerr << format("CONTACT_WITH_OTHER: {}\n",
                 features[CONTACT_WITH_OTHER]);  
  getline(cin, _);
#endif

  int delta = yolah.score(Yolah::BLACK) - yolah.score(Yolah::WHITE);
  int effective_delta = delta - (yolah.current_player() == Yolah::BLACK ? 0 : 1);
  features[SURE_WIN_BLACK] = effective_delta >= 1;
  features[SURE_WIN_WHITE] = effective_delta <= -1;

  features[FREE] = popcount(free_bb);
#ifndef NDEBUG
  cerr << "Free\n";
  cerr << yolah << '\n';
  cerr << format("FREE: {}\n", features[FREE]);
  getline(cin, _);
#endif
  features[TURN] = yolah.current_player();
}

void generate_features(const std::filesystem::path &input_file,
                       const std::filesystem::path &output_file) {
  using namespace std;
  auto size = filesystem::file_size(input_file);
  vector<uint8_t> encoding(size);
  ifstream ifs(input_file, ios::binary);
  ifs.read(reinterpret_cast<char *>(encoding.data()), size);
  stringbuf buffer;
  ostream os(&buffer);
  vector<Move> moves(Yolah::MAX_NB_PLIES);
  size_t n = 0;
  array<uint8_t, YolahFeatures::NB_FEATURES> game_features;
  while (n < size) {
    int nb_moves, nb_random_moves, black_score, white_score;
    data::decode_game(encoding.data() + n, moves, nb_moves, nb_random_moves,
                      black_score, white_score);
    uint8_t result = 0;
    if (white_score > black_score)
      result = 2;
    else if (white_score == black_score)
      result = 1;
    Yolah yolah;
    int i = 0;
    for (; i < nb_random_moves && !yolah.game_over(); i++) {
      yolah.play(moves[i]);
    }
    while (true) {
      set_features(game_features.data(), yolah);
      os.write(reinterpret_cast<const char *>(game_features.data()), NB_FEATURES);
      os << result;
      if (yolah.game_over()) break;
      yolah.play(moves[i++]);
    }
    n += 2 + nb_moves * 2 + 2;
  }
  ofstream ofs(output_file, ios::binary);
  ofs << buffer.str();
}

void encode_data(const std::string &src_dir, const std::string &dst_dir) {
  using namespace std;
  const filesystem::path src(src_dir);
  const filesystem::path dst(dst_dir);
  regex re_games("^games((?!.*features.*))",
                 regex_constants::ECMAScript | regex_constants::multiline);
  vector<filesystem::path> files_to_process;
  for (auto const &dir_entry : filesystem::directory_iterator(src_dir)) {
    auto path = dir_entry.path();
    if (!regex_search(path.filename().string(), re_games))
      continue;
    files_to_process.push_back(path);
  }
  for (const auto &path : files_to_process) {
    cout << "Processing: " << path << endl;
    auto output_filename =
        dst_dir /
        filesystem::path(path.filename()).replace_extension("features.txt");
    generate_features(path, output_filename);
    std::cout << "Done." << std::endl;
    break;
  }
}
} // namespace YolahFeatures
