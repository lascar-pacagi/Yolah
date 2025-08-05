#include "logic_net_learning.h"
#include <vector>
#include <regex>
#include <fstream>
#include <tuple>

LogicNetLearning::Builder& LogicNetLearning::Builder::set_crossover_rate(float r) {
    crossover_rate = r;
    return *this;
}

LogicNetLearning::Builder& LogicNetLearning::Builder::set_mutation_rate(float r) {
    mutation_rate = r;
    return *this;
}

LogicNetLearning::Builder& LogicNetLearning::Builder::set_selection_rate(float r) {
    selection_rate = r;
    return *this;
}

LogicNetLearning::Builder& LogicNetLearning::Builder::set_network_depth(int d) {
    network_depth = d;
    return *this;
}

LogicNetLearning::Builder& LogicNetLearning::Builder::set_population_size(int size) {
    assert(size % 2 == 0);
    population_size = size;
    return *this;
}

LogicNetLearning::Builder& LogicNetLearning::Builder::set_nb_iterations(int n) {
    nb_iterations = n;
    return *this;
}

LogicNetLearning::Builder& LogicNetLearning::Builder::set_training_data_path(const std::filesystem::path& p) {
    training_data_path = p;
    return *this;
}

LogicNetLearning::Builder& LogicNetLearning::Builder::set_logic_net_checkpoint_path(const std::filesystem::path& p) {
    logic_net_checkpoint_path = p;
    return *this;
}

LogicNetLearning LogicNetLearning::Builder::build() const {
    return { crossover_rate, mutation_rate, selection_rate, network_depth, population_size, nb_iterations, training_data_path, logic_net_checkpoint_path };
};

namespace {
    struct Data {
        Yolah yolah;
        int res; // 0 black victory, 1 draw, 2 white victory
    };
    std::tuple<std::vector<Data>, float, float, float> create_training_data(const std::filesystem::path& path) {
        using namespace std;
        vector<Data> data;
        ifstream ifs(path);
        regex re_nb_random(R"(.*_(\d+)r.*)", regex_constants::ECMAScript);
        smatch sm;
        string filename = path.filename().string(); 
        if (!regex_match(filename, sm, re_nb_random)) {
            throw "bad filename for game moves record";
        }
        int nb_random_moves = stoi(sm[1].str());
        //cout << nb_random_moves << '\n';
        regex re_moves(R"(((\w\d):(\w\d))+)", regex_constants::ECMAScript);
        regex re_scores(R"((\d+)/(\d+))", regex_constants::ECMAScript);
        int nb_positions = 0;
        int results[3]{};
        while (ifs) {
            int k = 0;
            Yolah yolah;
            string line; 
            getline(ifs, line);
            if (line == "") continue;
            smatch match;
            regex_search(line, match, re_scores);
            int black_score = atoi(match[1].str().c_str());
            int white_score = atoi(match[2].str().c_str());
            int match_result = black_score > white_score ? 0 : (white_score > black_score ? 2 : 1);            
            //cout << "(" << black_score << '/' << white_score << ")\n";
            for (auto it = sregex_iterator(begin(line), end(line), re_moves); it != sregex_iterator(); ++it) {
                if (k >= nb_random_moves) {
                    data.emplace_back(yolah, match_result);
                    results[match_result]++;
                    nb_positions++;
                }
                smatch match = *it;
                Square sq1 = make_square(match[2].str());
                Square sq2 = make_square(match[3].str());
                Move m(sq1, sq2);
                yolah.play(m);
                k++;
            }
        }
        float b_proportion = (float)results[0] / nb_positions;
        float d_proportion = (float)results[1] / nb_positions;
        float w_proportion = (float)results[2] / nb_positions;
        cout << "black: " << b_proportion << '\n' 
            << "draw: " << d_proportion << '\n'
            << "white: " << w_proportion << '\n';
        return { data, 1 / b_proportion, 1 / d_proportion, 1 / w_proportion };
    }
    float accuracy(const std::vector<Data>& training_data, const LogicNet& net) {
        int res = 0;
        for (const auto& [yolah, match_result] : training_data) {
            const auto [b, d, w] = net.forward(yolah);
            int prediction = 0;
            if (d > b && d > w) prediction = 1;
            if (w > b && w > d) prediction = 2;
            res += prediction == match_result;
        }
        return (float)res / training_data.size();
    }
    float compute_fitness(const std::vector<Data>& training_data, const float coeff_black, const float coeff_draw, 
                            const float coeff_white, const LogicNet& net) {
        // float res = 0;
        // constexpr float epsilon = std::numeric_limits<float>::min();
        // for (const auto& [yolah, match_result] : training_data) {
        //     float black_proba = 0, draw_proba = 0, white_proba = 0; 
        //     if (match_result == 0) black_proba = 1;
        //     else if (match_result == 1) draw_proba = 1;
        //     else white_proba = 1;
        //     const auto [b, d, w] = net.forward(yolah);
        //     int prediction = 0;
        //     if (d > b && d > w) prediction = 1;
        //     if (w > b && w > d) prediction = 2;
        //     res += prediction == match_result;
        //     res += black_proba * std::log(b + epsilon) + draw_proba * std::log(d + epsilon) + white_proba * std::log(w + epsilon);
        // }
        // return res / training_data.size();
        return accuracy(training_data, net);
    }
}

LogicNetLearning::LogicNetLearning(float crossover_rate, float mutation_rate, float selection_rate,
                                    int network_depth, int population_size, int nb_iterations,
                                    const std::filesystem::path& training_data_path, const std::filesystem::path& logic_net_checkpoint_path) 
                                    : crossover_rate(crossover_rate), mutation_rate(mutation_rate), selection_rate(selection_rate),
                                        network_depth(network_depth), population_size(population_size), nb_iterations(nb_iterations),
                                        training_data_path(training_data_path), logic_net_checkpoint_path(logic_net_checkpoint_path),
                                        fittest(network_depth), best_fitness(std::numeric_limits<float>::lowest()),
                                        population_fitness(population_size)
{
    using namespace std;
    const auto [training_data, coeff_black, coeff_draw, coeff_white] = create_training_data(training_data_path);
    cout << "Training data size: " << training_data.size() << '\n';
    // string _;
    // for (const auto& [yolah, black_proba, draw_proba, white_proba] : training_data) {
    //     cout << yolah << '\n';
    //     cout << black_proba << '\n';
    //     cout << draw_proba << '\n';
    //     cout << white_proba << "\n\n"; 
    //     getline(cin, _);       
    // }    
    int selection_size = max(2, int(population_size * selection_rate));
    mt19937_64 mt(chrono::system_clock::now().time_since_epoch().count());
    uniform_int_distribution<int> d_sel(0, population_size - 1);    
    uniform_real_distribution<float> d_cross(0, 1);
    uniform_real_distribution<float> d_mut(0, 1);
    uniform_int_distribution<int> d_swap(0, 1);
    uniform_int_distribution<uint8_t> d_gate(0, 15);
    uniform_int_distribution<uint16_t> d_input(0, LogicNet::Layer::SIZE * 2 - 1);
    auto selection = [&]{
        int best_index = 0;
        float best_value = numeric_limits<float>::lowest();
        for (int i = 0; i < selection_size; i++) {
            int idx = d_sel(mt);
            float v = population_fitness[idx]; 
            if (v > best_value) {
                best_index = idx;
                best_value = v;
            }
        }
        return population[best_index];
    };
    auto crossover = [&](LogicNet& net1, LogicNet& net2) {
        for (size_t i = 0; i < net1.layers.size(); i++) {
            if (d_swap(mt)) {
                swap(net1.layers[i], net2.layers[i]);
            }
        }
    };
    auto mutation = [&](LogicNet& net) {
        for (auto& l : net.layers) {
            for (int i = 0; i < LogicNet::Layer::SIZE; i++) {
                if (d_mut(mt) < mutation_rate) {
                    l.inputs1[i] = d_input(mt);
                }
                if (d_mut(mt) < mutation_rate) {
                    l.inputs2[i] = d_input(mt);
                }
                if (d_mut(mt) < mutation_rate) {
                    l.gates[i] = d_gate(mt);
                }
            }
        }
    };
    if (filesystem::exists(logic_net_checkpoint_path)) {
        ifstream ifs(logic_net_checkpoint_path);
        LogicNet net = LogicNet::from_json(ifs);
        for (int i = 0; i < population_size; i++) {
            LogicNet tmp = net;
            mutation(tmp);
            population.push_back(tmp);
        }
        population[0] = net;
    } else {
        for (int i = 0; i < population_size; i++) {
            population.emplace_back(network_depth);
        }
    }    
    vector<LogicNet> new_generation(population_size);
    for (int i = 0; i < nb_iterations; i++) {
        cout << i << endl;
        #pragma omp parallel for
        for (int i = 0; i < population_size; i++) {
            population_fitness[i] = compute_fitness(training_data, coeff_black, coeff_draw, coeff_white, population[i]);
        }
        float old_best_fitness = best_fitness;
        for (int i = 0; i < population_size; i++) {
            float v = population_fitness[i];
            if (v > best_fitness) {
                best_fitness = v;
                fittest = population[i];
            }
        }
        if (best_fitness > old_best_fitness) {
            cout << "Generation " << i << " best fitness: " << best_fitness << " accuracy: " << accuracy(training_data, fittest) << endl;
            ofstream ofs(logic_net_checkpoint_path);
            ofs << fittest.to_json();
        }
        for (int i = 0; i < population_size; i += 2) {
            LogicNet net1 = selection();
            LogicNet net2 = selection();
            if (d_cross(mt) < crossover_rate) {
                crossover(net1, net2);
            }
            mutation(net1);
            mutation(net2);
            new_generation[i] = std::move(net1);
            new_generation[i + 1] = std::move(net2);
        }
        new_generation[0] = fittest;
        swap(population, new_generation);
    }
}

LogicNet LogicNetLearning::get_fittest() const {
    return fittest;
}

// int main() {
//     LogicNetLearning::Builder builder;
//     try {
//         builder
//         .set_population_size(100)
//         .set_nb_iterations(100000)
//         .set_network_depth(10)
//         .set_crossover_rate(0.4)
//         .set_mutation_rate(0.01)
//         .set_logic_net_checkpoint_path("model.txt")
//         .set_training_data_path("data/games_2r_1s.txt")
//         .build();
//     } catch (const char* e) {
//         std::cout << e << '\n';
//     }
// }
