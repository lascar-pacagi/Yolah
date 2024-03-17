#ifndef MCTS_PLAYER_H
#define MCTS_PLAYER_H
#include "player.h"
#include <atomic>
#include <memory_resource>
#include "BS_thread_pool.h"
#include <numbers>

class MCTSPlayer : public Player {
    struct Node {
        std::atomic<uint32_t> nb_visits;
        std::atomic<float> value;
        Move action;
        std::pmr::vector<Node*> nodes;
        
        using allocator_type = std::pmr::polymorphic_allocator<>;

        explicit Node(Move action = Move::none(), allocator_type alloc = {}) : nb_visits(0), value(0), action(action), nodes(alloc) {            
        }
        bool is_leaf() const;
        void expand(Yolah);
        void update(float);
        uint32_t select() const;
    };
    static constexpr uint32_t VIRTUAL_LOSS = 4;
    static constexpr uint32_t NB_VISITS_BEFORE_EXPANSION = 8;
    const uint64_t thinking_time;
    Node root;
    BS::thread_pool pool;
    float playout(Yolah) const;
    void think(Yolah);
public:
    explicit MCTSPlayer(uint64_t microseconds, std::pmr::polymorphic_allocator<> alloc = {})
        : MCTSPlayer(microseconds, std::thread::hardware_concurrency(), alloc) { 
    }
    explicit MCTSPlayer(uint64_t microseconds, std::size_t nb_threads, std::pmr::polymorphic_allocator<> alloc = {}) 
        : thinking_time(microseconds), pool(nb_threads), root(Move::none(), alloc) {        
    } 
    Move play(Yolah) override;
};

#endif