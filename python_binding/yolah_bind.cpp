#include <pybind11/pybind11.h>
#include "game.h"
#include "zobrist.h"
#include "magic.h"
#include "heuristic.h"
#include <sstream>

namespace py = pybind11;

PYBIND11_MODULE(yolah_bind, m) {
    
    {
        magic::init();
        zobrist::init();
    }

    // py::class_<Yolah>(m, "Yolah")
    //     .def(py::init())
    //     .def("game_over", &Yolah::game_over)
    //     .def("nb_plies", &Yolah::nb_plies)
    //     .def("bitboard", &Yolah::bitboard)
    //     .def("free_squares", &Yolah::free_squares)
    //     .def("occupied_squares", &Yolah::occupied_squares)
    //     .def("__str__", 
    //         [](const Yolah& yolah) {
    //             std::stringstream ss;
    //             ss << yolah;
    //             return ss.str();
    //         });
    
    m.def("heuristic", 
          [](uint8_t player, 
            uint64_t black, 
            uint64_t white,
            uint64_t empty,
            uint16_t black_score,
            uint16_t white_score,
            uint16_t ply) {
                Yolah yolah;
                yolah.set_state(black, white, empty, black_score, white_score, ply); 
                return heuristic::evaluation(player, yolah);
            }, 
            "Evaluate a position");
}
