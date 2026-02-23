// Yolah board game -- ncurses UI
// Compile: g++ -std=c++23 -O3 -o yolah_ncurses yolah_ncurses.cpp -lncurses

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <clocale>
#include <ncurses.h>
#include <random>
#include <string_view>
#include <utility>
#include <vector>
#include <immintrin.h>

using namespace std;

// ─── Types ──────────────────────────────────────────────────────────

enum Square : int8_t {
    SQ_A1, SQ_B1, SQ_C1, SQ_D1, SQ_E1, SQ_F1, SQ_G1, SQ_H1,
    SQ_A2, SQ_B2, SQ_C2, SQ_D2, SQ_E2, SQ_F2, SQ_G2, SQ_H2,
    SQ_A3, SQ_B3, SQ_C3, SQ_D3, SQ_E3, SQ_F3, SQ_G3, SQ_H3,
    SQ_A4, SQ_B4, SQ_C4, SQ_D4, SQ_E4, SQ_F4, SQ_G4, SQ_H4,
    SQ_A5, SQ_B5, SQ_C5, SQ_D5, SQ_E5, SQ_F5, SQ_G5, SQ_H5,
    SQ_A6, SQ_B6, SQ_C6, SQ_D6, SQ_E6, SQ_F6, SQ_G6, SQ_H6,
    SQ_A7, SQ_B7, SQ_C7, SQ_D7, SQ_E7, SQ_F7, SQ_G7, SQ_H7,
    SQ_A8, SQ_B8, SQ_C8, SQ_D8, SQ_E8, SQ_F8, SQ_G8, SQ_H8,
    SQ_NONE, SQUARE_ZERO = 0, SQUARE_NB = 64
};

enum File : uint8_t { FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H };
enum Rank : uint8_t { RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8 };

enum Direction : int8_t {
    NORTH = 8, EAST = 1, SOUTH = -8, WEST = -1,
    NORTH_EAST = 9, SOUTH_EAST = -7, SOUTH_WEST = -9, NORTH_WEST = 7
};

enum MoveType { ORTHOGONAL, DIAGONAL };

constexpr uint8_t BLACK = 0, WHITE = 1, HOLE = 2, FREE = 3;

// ─── Bitboard constants ─────────────────────────────────────────────

constexpr uint64_t FileABB = 0x0101010101010101;
constexpr uint64_t FileHBB = FileABB << 7;
constexpr uint64_t Rank1BB = 0xFF;
constexpr uint64_t Rank8BB = Rank1BB << 56;

constexpr uint64_t BLACK_INITIAL_POSITION =
    0b10000000'00000000'00000000'00001000'00010000'00000000'00000000'00000001;
constexpr uint64_t WHITE_INITIAL_POSITION =
    0b00000001'00000000'00000000'00010000'00001000'00000000'00000000'10000000;

// ─── Bitboard utilities ─────────────────────────────────────────────

constexpr Square square_of(int i, int j) { return Square(i * 8 + j); }
constexpr File file_of(Square s) { return File(s & 7); }
constexpr Rank rank_of(Square s) { return Rank(s >> 3); }
constexpr uint64_t rank_bb(Rank r) { return Rank1BB << (8 * r); }
constexpr uint64_t rank_bb(Square s) { return rank_bb(rank_of(s)); }
constexpr uint64_t file_bb(File f) { return FileABB << f; }
constexpr uint64_t file_bb(Square s) { return file_bb(file_of(s)); }
constexpr uint64_t square_bb(Square s) { return 1ULL << s; }
constexpr Square operator+(Square s, Direction d) { return Square(int(s) + int(d)); }
Square& operator++(Square& d) { return d = Square(int(d) + 1); }
constexpr Square lsb(uint64_t b) { return Square(countr_zero(b)); }
Square pop_lsb(uint64_t& b) { Square s = lsb(b); b &= b - 1; return s; }

constexpr int manhattan_distance(Square sq1, Square sq2) {
    return abs(rank_of(sq1) - rank_of(sq2)) + abs(file_of(sq1) - file_of(sq2));
}

template<Direction D>
constexpr uint64_t shift(uint64_t b) {
    if constexpr (D == NORTH)      return b << 8;
    if constexpr (D == SOUTH)      return b >> 8;
    if constexpr (D == EAST)       return (b & ~FileHBB) << 1;
    if constexpr (D == WEST)       return (b & ~FileABB) >> 1;
    if constexpr (D == NORTH_EAST) return (b & ~FileHBB) << 9;
    if constexpr (D == NORTH_WEST) return (b & ~FileABB) << 7;
    if constexpr (D == SOUTH_EAST) return (b & ~FileHBB) >> 7;
    if constexpr (D == SOUTH_WEST) return (b & ~FileABB) >> 9;
    return 0;
}

constexpr bool is_ok(Square s) { return s >= SQ_A1 && s <= SQ_H8; }

// ─── Magic bitboards ────────────────────────────────────────────────

uint64_t reachable_squares(MoveType mt, Square sq, uint64_t occupied) {
    uint64_t moves = 0;
    Direction o_dir[4] = {NORTH, SOUTH, EAST, WEST};
    Direction d_dir[4] = {NORTH_EAST, SOUTH_EAST, SOUTH_WEST, NORTH_WEST};
    for (Direction d : (mt == ORTHOGONAL ? o_dir : d_dir)) {
        Square s = sq;
        while (true) {
            Square to = s + d;
            if (!is_ok(to) || manhattan_distance(s, to) > 2) break;
            uint64_t bb = square_bb(to);
            if (bb & occupied) break;
            moves |= bb;
            s = to;
        }
    }
    return moves;
}

struct Magic {
    uint64_t mask, magic;
    uint64_t* moves;
    uint32_t shift;
    uint32_t index(uint64_t occupied) const {
        return uint32_t(((occupied & mask) * magic) >> shift);
    }
};

uint64_t orthogonalTable[102400], diagonalTable[5248];
Magic orthogonalMagics[SQUARE_NB], diagonalMagics[SQUARE_NB];

uint64_t moves_bb(Square sq, uint64_t occupied) {
    return orthogonalMagics[sq].moves[orthogonalMagics[sq].index(occupied)]
         | diagonalMagics[sq].moves[diagonalMagics[sq].index(occupied)];
}

void init_magics(MoveType mt, uint64_t table[], Magic magics[]) {
    static constexpr uint64_t O_MAGIC[64] = {
        0x80011040002082,0x40022002100040,0x1880200081181000,0x2080240800100080,
        0x8080024400800800,0x4100080400024100,0xc080028001000a00,0x80146043000080,
        0x8120802080034004,0x8401000200240,0x202001282002044,0x81010021000b1000,
        0x808044000800,0x300080800c000200,0x8c000268411004,0x810080058020c100,
        0xc248608010400080,0x30024040002000,0x9001010042102000,0x210009001002,
        0xa0061d0018001100,0x2410808004000600,0x6400240008025001,0xc10600010340a4,
        0x628080044011,0x4810014040002000,0x380200080801000,0x10018580080010,
        0x101040080180180,0x9208020080040080,0x10400a21008,0x6800104200010484,
        0x21400280800020,0x9400402008401001,0x8430006800200400,0x8104411202000820,
        0x8010171000408,0x1202000402001008,0x881100904002208,0x15a0800a49802100,
        0x224001808004,0x4420201002424000,0xc04500020008080,0x2503009004210008,
        0x42801010010,0x2000400090100,0x8080011810040002,0x44401c008046000d,
        0x4000800521104100,0x82000b080400080,0x10821022420200,0x9488a82104100100,
        0x1004800041100,0x81600a0034008080,0xa00056210280400,0x5124088200,
        0x4210410010228202,0x1802230840001081,0x1002102000400901,0x1100c46010000901,
        0x281000408001003,0xc001001c00028809,0x10020008008c4102,0x280005008c014222,
    };
    static constexpr uint64_t D_MAGIC[64] = {
        0x811100100408200,0x412100401044020,0x404044c00408002,0xa0c070200010102,
        0x104042001400008,0x8802013008080000,0x1001008860080080,0x20220044202800,
        0x2002610802080160,0x4080800808610,0x91c2800a10a0132,0x400242401822000,
        0x8530040420040001,0x142010c210048,0x8841820801241004,0x804212084108801,
        0x2032402094100484,0x40202110010210a2,0x8010000800202020,0x800240421a800,
        0x62200401a00444,0x224082200820845,0x106021492012000,0x8481020082849000,
        0x40a110c59602800,0x10020108020400,0x208c020844080010,0x2000480004012020,
        0x8001004004044000,0xa044104128080200,0x1108008015cc1400,0x8284004801844400,
        0x8180a020c2004,0x9101004080100,0x8840264108800c0,0xc004200900200900,
        0x8040008020020020,0x20010802e1920200,0x80204000480a0,0xc0a80a100008400,
        0x4018808114000,0x90092200b9000,0x80020c0048000400,0x6018005500,
        0x80a0204110a00,0x4018808407201,0x6050040806500280,0x108208400c40180,
        0x803081210840480,0x201210402200200,0x200010400920042,0x902000a884110010,
        0x851002021004,0x43c08020120,0x6140500501010044,0x200a04440400c028,
        0x14a002084046000,0x10002409041040,0x100022020500880b,0x1000000000460802,
        0x21084104410,0x8000001053300104,0x4000182008c20048,0x112088105020200,
    };
    int size = 0;
    vector<uint64_t> occupancies, possible_moves;
    for (Square sq = SQ_A1; sq <= SQ_H8; ++sq) {
        occupancies.clear();
        possible_moves.clear();
        Magic& m = magics[sq];
        uint64_t edges = ((Rank1BB | Rank8BB) & ~rank_bb(sq)) |
                         ((FileABB | FileHBB) & ~file_bb(sq));
        uint64_t mb = reachable_squares(mt, sq, 0) & ~edges;
        m.mask = mb;
        m.shift = 64 - popcount(m.mask);
        m.magic = (mt == ORTHOGONAL ? O_MAGIC : D_MAGIC)[sq];
        m.moves = table + size;
        uint64_t b = 0;
        do {
            occupancies.push_back(b);
            possible_moves.push_back(reachable_squares(mt, sq, b));
            b = (b - mb) & mb;
            size++;
        } while (b);
        for (size_t j = 0; j < occupancies.size(); j++) {
            int32_t idx = m.index(occupancies[j]);
            m.moves[idx] = possible_moves[j];
        }
    }
}

void init_all_magics() {
    init_magics(ORTHOGONAL, orthogonalTable, orthogonalMagics);
    init_magics(DIAGONAL, diagonalTable, diagonalMagics);
}

// ─── Move ───────────────────────────────────────────────────────────

class Move {
    uint16_t data = 0;
public:
    constexpr Move() = default;
    constexpr Move(Square from, Square to) : data((to << 6) + from) {}
    constexpr Square from_sq() const { return Square(data & 0x3F); }
    constexpr Square to_sq() const { return Square((data >> 6) & 0x3F); }
    static constexpr Move none() { return Move(); }
    constexpr bool operator==(const Move& m) const { return data == m.data; }
    constexpr bool operator!=(const Move& m) const { return data != m.data; }
};

static constexpr uint16_t MAX_NB_MOVES = 75;

class MoveList {
    Move move_list[MAX_NB_MOVES], *last = move_list;
public:
    const Move* begin() const { return move_list; }
    const Move* end() const { return last; }
    size_t size() const { return last - move_list; }
    const Move& operator[](size_t i) const { return move_list[i]; }
    friend class Yolah;
};

// ─── Yolah ──────────────────────────────────────────────────────────

class Yolah {
    uint64_t black = BLACK_INITIAL_POSITION;
    uint64_t white = WHITE_INITIAL_POSITION;
    uint64_t holes = 0;
    uint8_t black_score = 0, white_score = 0, ply = 0;
public:
    bool game_over() const {
        uint64_t possible = ~holes & ~black & ~white;
        uint64_t players = black | white;
        uint64_t around = shift<NORTH>(players) | shift<SOUTH>(players) |
            shift<EAST>(players) | shift<WEST>(players) |
            shift<NORTH_EAST>(players) | shift<NORTH_WEST>(players) |
            shift<SOUTH_EAST>(players) | shift<SOUTH_WEST>(players);
        return (around & possible) == 0;
    }

    uint8_t current_player() const { return ply & 1; }
    pair<uint8_t, uint8_t> score() const { return {black_score, white_score}; }

    uint8_t get(Square sq) const {
        uint64_t bb = square_bb(sq);
        if (holes & bb) return HOLE;
        if (black & bb) return BLACK;
        if (white & bb) return WHITE;
        return FREE;
    }

    uint8_t get(int i, int j) const { return get(square_of(i, j)); }

    void moves(MoveList& ml) const {
        Move* p = ml.move_list;
        uint64_t occupied = black | white | holes;
        uint64_t bb = (ply & 1) ? white : black;
        while (bb) {
            Square from = pop_lsb(bb);
            uint64_t b = moves_bb(from, occupied) & ~occupied;
            while (b) *p++ = Move(from, pop_lsb(b));
        }
        if (p == ml.move_list) *p++ = Move::none();
        ml.last = p;
    }

    void play(Move m) {
        if (m != Move::none()) {
            uint64_t p1 = square_bb(m.from_sq()), p2 = square_bb(m.to_sq());
            if (ply & 1) { white ^= p1 | p2; white_score++; }
            else         { black ^= p1 | p2; black_score++; }
            holes |= p1;
        }
        ply++;
    }

    void undo(Move m) {
        ply--;
        if (m != Move::none()) {
            uint64_t p1 = square_bb(m.from_sq()), p2 = square_bb(m.to_sq());
            if (ply & 1) { white ^= p1 | p2; white_score--; }
            else         { black ^= p1 | p2; black_score--; }
            holes ^= p1;
        }
    }

    Move random_move(mt19937& rng) const {
        MoveList ml;
        moves(ml);
        uniform_int_distribution<size_t> d(0, ml.size() - 1);
        return ml[d(rng)];
    }
};

// ─── Square name ────────────────────────────────────────────────────

string_view square_name(Square sq) {
    static constexpr string_view names[64] = {
        "a1","b1","c1","d1","e1","f1","g1","h1",
        "a2","b2","c2","d2","e2","f2","g2","h2",
        "a3","b3","c3","d3","e3","f3","g3","h3",
        "a4","b4","c4","d4","e4","f4","g4","h4",
        "a5","b5","c5","d5","e5","f5","g5","h5",
        "a6","b6","c6","d6","e6","f6","g6","h6",
        "a7","b7","c7","d7","e7","f7","g7","h7",
        "a8","b8","c8","d8","e8","f8","g8","h8",
    };
    return names[sq];
}

// ─── ncurses UI ─────────────────────────────────────────────────────

enum ColorPair {
    CP_DEFAULT = 1, CP_BLACK_PIECE, CP_WHITE_PIECE,
    CP_HOLE, CP_CURSOR, CP_SELECTED, CP_REACHABLE,
    CP_LAST_MOVE, // highlight the from/to squares of the last move played
    CP_STATUS, CP_TITLE,
    // For the move list panel on the right
    CP_BLACK_PIECE_TEXT, CP_WHITE_PIECE_TEXT
};

void init_colors() {
    start_color();        // enable color support in ncurses
    // assume_default_colors(fg, bg) overrides the terminal's default colors,
    // forcing a black background regardless of the terminal theme
    assume_default_colors(COLOR_WHITE, COLOR_BLACK);

    // init_pair(id, foreground, background) defines a color pair that
    // can later be activated with attron(COLOR_PAIR(id))
    init_pair(CP_DEFAULT,      COLOR_WHITE,   COLOR_BLACK);
    init_pair(CP_BLACK_PIECE,  COLOR_BLUE,    COLOR_BLACK);
    init_pair(CP_WHITE_PIECE,  COLOR_WHITE,   COLOR_BLACK);
    init_pair(CP_HOLE,         COLOR_BLACK,   COLOR_BLACK);
    init_pair(CP_CURSOR,       COLOR_BLACK,   COLOR_YELLOW);
    init_pair(CP_SELECTED,     COLOR_BLACK,   COLOR_GREEN);
    init_pair(CP_REACHABLE,    COLOR_BLACK,   COLOR_CYAN);
    // A_REVERSE swaps foreground and background for a given color pair,
    // but a dedicated pair with an explicit background is more readable
    init_pair(CP_LAST_MOVE,    COLOR_WHITE,   COLOR_MAGENTA);
    init_pair(CP_STATUS,       COLOR_YELLOW,  COLOR_BLACK);
    init_pair(CP_TITLE,        COLOR_GREEN,   COLOR_BLACK);

    // For the move list panel on the right (black background)
    init_pair(CP_BLACK_PIECE_TEXT, COLOR_BLUE,  COLOR_BLACK);
    init_pair(CP_WHITE_PIECE_TEXT, COLOR_WHITE, COLOR_BLACK);
}

// Board drawing constants
constexpr int BOARD_X = 2;
constexpr int BOARD_Y = 2;
constexpr int CELL_W = 4;
constexpr int CELL_H = 2;

// Draw a horizontal border line using ACS (Alternate Character Set) macros.
// ACS_* are ncurses constants for portable box-drawing characters that
// render correctly on any terminal, unlike raw UTF-8 which can cause
// column misalignment with the basic ncurses byte-oriented functions.
void draw_hline(int row, int col, chtype left, chtype mid, chtype right) {
    move(row, col);       // move cursor to (row, col)
    addch(left);          // addch prints a single character (supports ACS_*)
    for (int i = 0; i < 8; i++) {
        addch(ACS_HLINE); addch(ACS_HLINE); addch(ACS_HLINE);
        if (i < 7) addch(mid);
    }
    addch(right);
}

// Column where the move history panel starts (right of the board)
constexpr int PANEL_X = BOARD_X + 2 + 8 * CELL_W + 3;

void draw_board(const Yolah& game, Square cursor, Square selected,
                const vector<Square>& reachable, const vector<Move>& history,
                const string& message) {
    erase(); // clear the virtual screen (changes are not visible until refresh())

    // Title
    auto [bs, ws] = game.score();
    const char* player_name = game.current_player() == BLACK ? "Black" : "White";
    // attron/attroff turn text attributes on/off for subsequent output
    // A_BOLD makes text bold, COLOR_PAIR(id) selects a color pair
    attron(COLOR_PAIR(CP_TITLE) | A_BOLD);
    // mvprintw(row, col, fmt, ...) moves to (row,col) and prints formatted text
    mvprintw(0, BOARD_X, "Yolah");
    attroff(A_BOLD);
    mvprintw(0, BOARD_X + 8, "- %s's turn", player_name);
    attroff(COLOR_PAIR(CP_TITLE));
    attron(COLOR_PAIR(CP_DEFAULT));
    mvprintw(0, BOARD_X + 30, "Score: %d/%d", bs, ws);
    attroff(COLOR_PAIR(CP_DEFAULT));

    // Last move: highlight its from-square (now a hole) and to-square
    Move last_move = history.empty() ? Move::none() : history.back();

    // Top border using ACS macros
    draw_hline(BOARD_Y, BOARD_X + 2, ACS_ULCORNER, ACS_TTEE, ACS_URCORNER);

    for (int rank = 7; rank >= 0; rank--) {
        int row = BOARD_Y + 1 + (7 - rank) * CELL_H;

        // Rank label
        mvprintw(row, BOARD_X, "%d", rank + 1);

        for (int file = 0; file < 8; file++) {
            int col = BOARD_X + 2 + file * CELL_W;
            Square sq = square_of(rank, file);
            uint8_t content = game.get(sq);

            // Determine cell symbol and color pair
            // @ for black, O for white, . for free
            int attr = COLOR_PAIR(CP_DEFAULT);
            char symbol = '.';

            if (content == BLACK) {
                symbol = '@';
                attr = COLOR_PAIR(CP_BLACK_PIECE) | A_BOLD;
            } else if (content == WHITE) {
                symbol = 'O';
                attr = COLOR_PAIR(CP_WHITE_PIECE) | A_BOLD;
            } else if (content == HOLE) {
                symbol = ' ';
                attr = COLOR_PAIR(CP_HOLE);
            }

            // Overlay priority (highest first): cursor > selected > reachable > last move
            bool is_reachable = find(reachable.begin(), reachable.end(), sq) != reachable.end();
            bool is_last_move = last_move != Move::none() &&
                (sq == last_move.from_sq() || sq == last_move.to_sq());
            if (sq == cursor) {
                attr = COLOR_PAIR(CP_CURSOR) | A_BOLD;
            } else if (sq == selected) {
                attr = COLOR_PAIR(CP_SELECTED) | A_BOLD;
            } else if (is_reachable) {
                attr = COLOR_PAIR(CP_REACHABLE) | A_BOLD;
            } else if (is_last_move) {
                // Highlight the from and to squares of the last move with
                // a magenta background so the player can see what just happened
                attr = COLOR_PAIR(CP_LAST_MOVE) | A_BOLD;
            }

            // mvaddch moves to position and prints one char (ACS_VLINE = │)
            mvaddch(row, col, ACS_VLINE);
            attron(attr);              // activate color/style for this cell
            addch(' ');                // padding
            addch(symbol);             // piece symbol
            addch(' ');                // padding
            attroff(attr);             // deactivate to avoid leaking into next cell
        }
        addch(ACS_VLINE);

        // Row separator
        if (rank > 0) {
            draw_hline(row + 1, BOARD_X + 2, ACS_LTEE, ACS_PLUS, ACS_RTEE);
        }
    }

    // Bottom border
    int bottom = BOARD_Y + 1 + 8 * CELL_H - 1;
    draw_hline(bottom, BOARD_X + 2, ACS_LLCORNER, ACS_BTEE, ACS_LRCORNER);
    // mvaddstr(row, col, str) moves to (row,col) and prints a string
    mvaddstr(bottom + 1, BOARD_X + 2, "  a   b   c   d   e   f   g   h");

    // ── Move history panel on the right ──
    attron(COLOR_PAIR(CP_TITLE) | A_BOLD);
    mvaddstr(BOARD_Y, PANEL_X, "Moves");
    attroff(COLOR_PAIR(CP_TITLE) | A_BOLD);

    // Display moves in two columns: move# black white
    // getmaxy(stdscr) returns the terminal height in rows
    int max_rows = getmaxy(stdscr) - BOARD_Y - 3;
    int total_turns = (int(history.size()) + 1) / 2;
    // If the history is longer than the visible area, show only the last moves
    int first_turn = total_turns > max_rows ? total_turns - max_rows : 0;

    for (int t = first_turn; t < total_turns; t++) {
        int row = BOARD_Y + 1 + (t - first_turn);
        int i = t * 2; // index of black's move

        // Move number
        attron(COLOR_PAIR(CP_DEFAULT));
        mvprintw(row, PANEL_X, "%2d.", t + 1);
        attroff(COLOR_PAIR(CP_DEFAULT));

        // Black's move
        attron(COLOR_PAIR(CP_BLACK_PIECE_TEXT));
        if (history[i] == Move::none())
            mvaddstr(row, PANEL_X + 4, "pass");
        else
            mvprintw(row, PANEL_X + 4, "%s-%s",
                string(square_name(history[i].from_sq())).c_str(),
                string(square_name(history[i].to_sq())).c_str());
        attroff(COLOR_PAIR(CP_BLACK_PIECE_TEXT));

        // White's move (if played)
        if (i + 1 < int(history.size())) {
            attron(COLOR_PAIR(CP_WHITE_PIECE_TEXT));
            if (history[i + 1] == Move::none())
                mvaddstr(row, PANEL_X + 12, "pass");
            else
                mvprintw(row, PANEL_X + 12, "%s-%s",
                    string(square_name(history[i + 1].from_sq())).c_str(),
                    string(square_name(history[i + 1].to_sq())).c_str());
            attroff(COLOR_PAIR(CP_WHITE_PIECE_TEXT));
        }
    }

    // Status line
    attron(COLOR_PAIR(CP_STATUS));
    mvprintw(bottom + 3, BOARD_X, "%s", message.c_str());
    attroff(COLOR_PAIR(CP_STATUS));

    // Help
    attron(COLOR_PAIR(CP_DEFAULT));
    mvaddstr(bottom + 5, BOARD_X, "Arrows: move  Enter: select  Esc: cancel  u: undo  q: quit");
    attroff(COLOR_PAIR(CP_DEFAULT));

    refresh(); // flush the virtual screen to the actual terminal
}

// ─── Main ───────────────────────────────────────────────────────────

int main() {
    setlocale(LC_ALL, ""); // needed for ncurses to handle the terminal's locale
    init_all_magics();

    // Mode selection (before ncurses)
    printf("Yolah - Select mode:\n");
    printf("  1. Human vs Human\n");
    printf("  2. Human vs Random AI\n");
    printf("Choice: ");
    fflush(stdout);
    int mode = 0;
    while (mode != 1 && mode != 2) {
        char c = getchar();
        if (c == '1') mode = 1;
        if (c == '2') mode = 2;
    }

    // Init ncurses
    initscr();                // initialize ncurses, takes over the terminal
    cbreak();                 // disable line buffering, pass keys immediately
    noecho();                 // don't echo typed characters to the screen
    keypad(stdscr, TRUE);     // enable arrow keys and special keys (KEY_UP, etc.)
    curs_set(0);              // hide the blinking terminal cursor
    init_colors();

    Yolah game;
    mt19937 rng(random_device{}());
    Square cursor = SQ_D4;
    Square selected = SQ_NONE;
    vector<Square> reachable;
    vector<Move> history;
    string message = "Select a piece to move.";

    auto compute_reachable = [&](Square sq) {
        reachable.clear();
        MoveList ml;
        game.moves(ml);
        for (auto it = ml.begin(); it != ml.end(); ++it) {
            if (it->from_sq() == sq) reachable.push_back(it->to_sq());
        }
    };

    bool running = true;
    while (running) {
        // AI turn
        if (mode == 2 && game.current_player() == WHITE && !game.game_over()) {
            Move m = game.random_move(rng);
            game.play(m);
            history.push_back(m);
            selected = SQ_NONE;
            reachable.clear();
            if (m != Move::none()) {
                message = "AI played ";
                message += square_name(m.from_sq());
                message += ":";
                message += square_name(m.to_sq());
                cursor = m.to_sq();
            } else {
                message = "AI passed.";
            }
        }

        if (game.game_over()) {
            auto [bs, ws] = game.score();
            if (bs > ws) message = "Game over! Black wins " + to_string(bs) + "-" + to_string(ws);
            else if (ws > bs) message = "Game over! White wins " + to_string(ws) + "-" + to_string(bs);
            else message = "Game over! Draw " + to_string(bs) + "-" + to_string(ws);
        }

        draw_board(game, cursor, selected, reachable, history, message);

        int ch = getch(); // block and wait for a single keypress
        int rank = rank_of(cursor), file = file_of(cursor);

        switch (ch) {
        case KEY_UP:    if (rank < 7) cursor = square_of(rank + 1, file); break;
        case KEY_DOWN:  if (rank > 0) cursor = square_of(rank - 1, file); break;
        case KEY_RIGHT: if (file < 7) cursor = square_of(rank, file + 1); break;
        case KEY_LEFT:  if (file > 0) cursor = square_of(rank, file - 1); break;

        case '\n':
        case KEY_ENTER:
            if (game.game_over()) break;
            if (selected == SQ_NONE) {
                // Select a piece
                uint8_t content = game.get(cursor);
                if (content == game.current_player()) {
                    selected = cursor;
                    compute_reachable(selected);
                    if (reachable.empty()) {
                        selected = SQ_NONE;
                        message = "This piece has no moves.";
                    } else {
                        message = "Select destination.";
                    }
                } else {
                    message = "Not your piece!";
                }
            } else {
                // Try to move
                if (find(reachable.begin(), reachable.end(), cursor) != reachable.end()) {
                    Move m(selected, cursor);
                    game.play(m);
                    history.push_back(m);
                    selected = SQ_NONE;
                    reachable.clear();
                    message = "Select a piece to move.";
                } else {
                    message = "Invalid destination.";
                }
            }
            break;

        case 27: // Escape
            selected = SQ_NONE;
            reachable.clear();
            message = "Select a piece to move.";
            break;

        case 'u':
        case 'U':
            if (!history.empty()) {
                // In mode 2, undo both AI and human moves
                if (mode == 2 && history.size() >= 2) {
                    game.undo(history.back()); history.pop_back();
                    game.undo(history.back()); history.pop_back();
                } else if (mode == 1 && !history.empty()) {
                    game.undo(history.back()); history.pop_back();
                }
                selected = SQ_NONE;
                reachable.clear();
                message = "Move undone.";
            }
            break;

        case 'p':
        case 'P':
            if (!game.game_over()) {
                MoveList ml;
                game.moves(ml);
                if (ml.size() == 1 && ml[0] == Move::none()) {
                    game.play(Move::none());
                    history.push_back(Move::none());
                    message = "Turn passed.";
                }
            }
            break;

        case 'q':
        case 'Q':
            running = false;
            break;
        }
    }

    endwin(); // restore the terminal to its normal state
    return 0;
}
