#ifndef LOGIC_NET_LEARNING_H
#define LOGIC_NET_LEARNING_H
#include "logic_net.h"
#include <filesystem>
#include <limits>
#include <vector>

struct LogicNetLearning {
    struct Builder {
        float crossover_rate = 0.4;
        float mutation_rate  = 0.01;
        float selection_rate = 0.05;
        int network_depth    = 8;
        int population_size  = 100;
        int nb_iterations    = 1000;
        std::filesystem::path training_data_path;
        std::filesystem::path logic_net_checkpoint_path;
        Builder& set_crossover_rate(float r);
        Builder& set_mutation_rate(float r);
        Builder& set_selection_rate(float r);
        Builder& set_network_depth(int d);
        Builder& set_population_size(int size);
        Builder& set_nb_iterations(int n);
        Builder& set_training_data_path(const std::filesystem::path& path);
        Builder& set_logic_net_checkpoint_path(const std::filesystem::path& path);
        LogicNetLearning build() const;
    };
    float crossover_rate;
    float mutation_rate;
    float selection_rate;
    int network_depth;
    int population_size;
    int nb_iterations;
    std::filesystem::path training_data_path;
    std::filesystem::path logic_net_checkpoint_path;
    LogicNet fittest;
    float best_fitness = std::numeric_limits<float>::lowest();
    std::vector<LogicNet> population;
    std::vector<float>    population_fitness;
    LogicNetLearning(float crossover_rate, float mutation_rate, float selection_rate,
                        int network_depth, int population_size, int nb_iterations,
                        const std::filesystem::path& training_data_path,
                        const std::filesystem::path& logic_net_checkpoint_path);
    LogicNet get_fittest() const;
};

#endif