#include "ffnn.h"
#include <iostream>
#include "yolah_features.h"
#include <string>
#include <filesystem>
#include <vector>

// g++ -std=c++2a -O3 -march=native -mavx2 -ffast-math -funroll-loops -I../player -I../game -I../misc ffnn.c
// int main(int argc, char* argv[]) {
//     using namespace std;
//     FFNN<YolahFeatures::NB_FEATURES, 128, 64, 3> net("features_128x64x3.quantized.txt");
//     alignas(64) uint8_t features[decltype(net)::I_PADDED]{};
//     const string path = argv[1];
//     auto size = filesystem::file_size(path);
//     vector<uint8_t> encoding(size);
//     ifstream ifs(path, ios::binary);
//     ifs.read(reinterpret_cast<char *>(encoding.data()), size);
//     size_t i = 0;
//     for (int i = 0; i < encoding.size(); i += YolahFeatures::NB_FEATURES + 1) {
//         memset(features, 0, sizeof(features));
//         for (int j = 0; j < YolahFeatures::NB_FEATURES; j++) {
//             features[j] = encoding[i + j];
//         }        
//         const auto [black_proba, draw_proba, white_proba] = net(features);
//         cout << setprecision(17) << black_proba << '\n';
//         cout << draw_proba << '\n';
//         cout << white_proba << "\n";    
//     }    
// }
