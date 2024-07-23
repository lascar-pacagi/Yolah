#ifndef MCTS_PLAYER_H
#define MCTS_PLAYER_H
#include "player.h"
#include <atomic>
#include <memory_resource>
#include "BS_thread_pool.h"
#include <numbers>

class MCTSPlayer : public Player {
    struct Node {
        static size_t NB_NODES;
        static constexpr float C = std::numbers::sqrt2_v<float>;
        std::atomic<uint32_t> nb_visits = 1;
        std::atomic<uint32_t> virtual_loss = 0;
        std::atomic<int32_t> value = 0;
        std::atomic<uint8_t> expanded = 0;
        Move action;
        std::pmr::vector<Node> nodes;
        
        using allocator_type = std::pmr::polymorphic_allocator<>;
        explicit Node(allocator_type alloc = {}) : Node(Move::none(), alloc) {            
        }
        explicit Node(Move action, allocator_type alloc = {}) : action(action), nodes(alloc) {
            ++Node::NB_NODES;
        }      
        Node(Node&& n, allocator_type) noexcept 
            : nb_visits(n.nb_visits.load()), virtual_loss(n.virtual_loss.load()), value(n.value.load()), 
              expanded(n.expanded.load()), action(std::move(n.action)), nodes(std::move(n.nodes)) {
        }    
        bool is_leaf() const;
        void expand(const Yolah&);
        void update(int32_t);
        uint32_t select() const;        
    };
    friend std::ostream& operator<<(std::ostream&, const Node&);

    static constexpr uint32_t VIRTUAL_LOSS = 4;
    static constexpr uint32_t NB_VISITS_BEFORE_EXPANSION = 8;
    const uint64_t thinking_time;
    Node root;
    BS::thread_pool pool;
    int32_t playout(Yolah) const;
    void think(Yolah);
    void reset();
public:
    explicit MCTSPlayer(uint64_t microseconds, std::pmr::polymorphic_allocator<> alloc = {})
        : MCTSPlayer(microseconds, std::thread::hardware_concurrency(), alloc) { 
    }
    explicit MCTSPlayer(uint64_t microseconds, std::size_t nb_threads, std::pmr::polymorphic_allocator<> alloc = {}) 
        : thinking_time(microseconds), root(Move::none(), alloc), pool(static_cast<BS::concurrency_t>(nb_threads)) {  
    } 
    uint64_t microseconds() const {
        return thinking_time;
    }
    size_t nb_threads() const {
        return pool.get_thread_count();
    }
    Move play(Yolah) override;
    std::string info() override;
    json config() override;
};

std::ostream& operator<<(std::ostream& os, const MCTSPlayer::Node& n);

#endif