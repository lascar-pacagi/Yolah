#include "ascii_observer.h"

void AsciiObserver::operator()(Yolah yolah) const {    
    std::cout << yolah;    
}

void AsciiObserver::operator()(uint8_t player, Move m) const {    
    std::cout << "[" << (player == Yolah::BLACK ? "black" : "white") << " move] " << m << std::endl;
}