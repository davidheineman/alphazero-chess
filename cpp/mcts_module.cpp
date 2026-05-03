#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "include.hpp"

namespace py = pybind11;
using namespace chess;

static constexpr float VIRTUAL_LOSS_VALUE = 3.0f;
static constexpr int ACTION_SPACE = 4672;  // 64 * 73
static constexpr int NUM_PLANES = 19;

// Direction tables matching Python encode.py
static const int QUEEN_DIRS[8][2] = {
    {-1, 0}, {1, 0}, {0, -1}, {0, 1},
    {-1, -1}, {-1, 1}, {1, -1}, {1, 1}
};
static const int KNIGHT_MOVES[8][2] = {
    {-2, -1}, {-2, 1}, {-1, -2}, {-1, 2},
    {1, -2}, {1, 2}, {2, -1}, {2, 1}
};
static const int UNDERPROMO_DIRS[3] = {-1, 0, 1};

static int move_to_action(const Move& move, const Board& board) {
    int from_sq = move.from().index();
    int to_sq = move.to().index();
    int from_r = from_sq / 8, from_c = from_sq % 8;
    int to_r = to_sq / 8, to_c = to_sq % 8;
    int dr = to_r - from_r, dc = to_c - from_c;

    // Underpromotion (not queen)
    if (move.typeOf() == Move::PROMOTION) {
        auto promo = move.promotionType();
        if (promo != PieceType::QUEEN) {
            int dir_idx = -1;
            int clamped = std::max(-1, std::min(1, dc));
            for (int i = 0; i < 3; i++) {
                if (UNDERPROMO_DIRS[i] == clamped) { dir_idx = i; break; }
            }
            int piece_idx = -1;
            if (promo == PieceType::KNIGHT) piece_idx = 0;
            else if (promo == PieceType::BISHOP) piece_idx = 1;
            else if (promo == PieceType::ROOK) piece_idx = 2;
            int move_type = 56 + 8 + dir_idx * 3 + piece_idx;
            return from_sq * 73 + move_type;
        }
    }

    // Knight move
    int adr = std::abs(dr), adc = std::abs(dc);
    if ((adr == 2 && adc == 1) || (adr == 1 && adc == 2)) {
        for (int i = 0; i < 8; i++) {
            if (KNIGHT_MOVES[i][0] == dr && KNIGHT_MOVES[i][1] == dc) {
                return from_sq * 73 + 56 + i;
            }
        }
    }

    // Queen-type move (includes queen promotions)
    if (dr != 0 || dc != 0) {
        int dir_r = (dr > 0) ? 1 : (dr < 0) ? -1 : 0;
        int dir_c = (dc > 0) ? 1 : (dc < 0) ? -1 : 0;
        int dir_idx = -1;
        for (int i = 0; i < 8; i++) {
            if (QUEEN_DIRS[i][0] == dir_r && QUEEN_DIRS[i][1] == dir_c) {
                dir_idx = i; break;
            }
        }
        int distance = std::max(adr, adc) - 1;
        return from_sq * 73 + dir_idx * 7 + distance;
    }

    return from_sq * 73;
}

static void encode_board(const Board& board, float* planes) {
    std::memset(planes, 0, NUM_PLANES * 64 * sizeof(float));

    // Piece planes (12)
    static const PieceType piece_types[6] = {
        PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP,
        PieceType::ROOK, PieceType::QUEEN, PieceType::KING
    };
    for (int pt = 0; pt < 6; pt++) {
        for (int color = 0; color < 2; color++) {
            int plane_idx = pt * 2 + color;
            auto c = (color == 0) ? Color::WHITE : Color::BLACK;
            Bitboard bb = board.pieces(piece_types[pt], c);
            while (bb) {
                int sq = bb.pop();
                planes[plane_idx * 64 + sq] = 1.0f;
            }
        }
    }

    // Side to move
    if (board.sideToMove() == Color::WHITE) {
        for (int i = 0; i < 64; i++) planes[12 * 64 + i] = 1.0f;
    }

    // Castling rights
    auto cr = board.castlingRights();
    using Side = Board::CastlingRights::Side;
    if (cr.has(Color::WHITE, Side::KING_SIDE))
        for (int i = 0; i < 64; i++) planes[13 * 64 + i] = 1.0f;
    if (cr.has(Color::WHITE, Side::QUEEN_SIDE))
        for (int i = 0; i < 64; i++) planes[14 * 64 + i] = 1.0f;
    if (cr.has(Color::BLACK, Side::KING_SIDE))
        for (int i = 0; i < 64; i++) planes[15 * 64 + i] = 1.0f;
    if (cr.has(Color::BLACK, Side::QUEEN_SIDE))
        for (int i = 0; i < 64; i++) planes[16 * 64 + i] = 1.0f;

    // En passant
    auto ep = board.enpassantSq();
    if (ep != Square::NO_SQ) {
        planes[17 * 64 + ep.index()] = 1.0f;
    }

    // Halfmove clock
    float hmc = board.halfMoveClock() / 100.0f;
    for (int i = 0; i < 64; i++) planes[18 * 64 + i] = hmc;
}


struct MCTSNode {
    Board board;
    MCTSNode* parent = nullptr;
    int action = -1;
    float prior = 0.0f;
    std::vector<std::unique_ptr<MCTSNode>> children;
    int visit_count = 0;
    float value_sum = 0.0f;
    bool is_expanded = false;
    int virtual_loss_count = 0;

    // For lazy expansion
    Move pending_move = Move::NO_MOVE;
    bool needs_board_init = false;

    MCTSNode() = default;
    MCTSNode(const Board& b) : board(b) {}

    float q_value() const {
        int total = visit_count + virtual_loss_count;
        if (total == 0) return 0.0f;
        return (value_sum - virtual_loss_count * VIRTUAL_LOSS_VALUE) / total;
    }

    float ucb_score(float c_puct, int parent_visits) const {
        int total = visit_count + virtual_loss_count;
        float exploration = c_puct * prior * std::sqrt((float)parent_visits) / (1 + total);
        return q_value() + exploration;
    }

    MCTSNode* best_child(float c_puct) {
        int pv = visit_count + virtual_loss_count;
        MCTSNode* best = nullptr;
        float best_score = -1e9f;
        for (auto& c : children) {
            float s = c->ucb_score(c_puct, pv);
            if (s > best_score) { best_score = s; best = c.get(); }
        }
        return best;
    }

    MCTSNode* select_leaf(float c_puct) {
        MCTSNode* node = this;
        while (node->is_expanded && !node->children.empty()) {
            node = node->best_child(c_puct);
        }
        // Lazy board init
        if (node->needs_board_init && node->parent) {
            node->board = node->parent->board;
            node->board.makeMove(node->pending_move);
            node->needs_board_init = false;
        }
        return node;
    }

    void expand(const float* policy) {
        is_expanded = true;
        Movelist moves;
        movegen::legalmoves(moves, board);

        // Build masked priors
        float total = 0.0f;
        std::vector<std::pair<int, float>> action_priors;
        action_priors.reserve(moves.size());

        for (int i = 0; i < (int)moves.size(); i++) {
            int a = move_to_action(moves[i], board);
            float p = policy[a];
            action_priors.push_back({a, p});
            total += p;
        }

        if (total > 0) {
            for (auto& [a, p] : action_priors) p /= total;
        } else {
            float uniform = 1.0f / action_priors.size();
            for (auto& [a, p] : action_priors) p = uniform;
        }

        children.reserve(moves.size());
        for (int i = 0; i < (int)moves.size(); i++) {
            auto child = std::make_unique<MCTSNode>();
            child->parent = this;
            child->action = action_priors[i].first;
            child->prior = action_priors[i].second;
            child->pending_move = moves[i];
            child->needs_board_init = true;
            children.push_back(std::move(child));
        }
    }

    void add_virtual_loss() {
        MCTSNode* node = this;
        while (node) { node->virtual_loss_count++; node = node->parent; }
    }

    void revert_virtual_loss() {
        MCTSNode* node = this;
        while (node) { node->virtual_loss_count--; node = node->parent; }
    }

    void backup(float value) {
        MCTSNode* node = this;
        while (node) {
            node->visit_count++;
            node->value_sum += value;
            value = -value;
            node = node->parent;
        }
    }
};


class MCTSSearch {
public:
    std::unique_ptr<MCTSNode> root;
    float c_puct;
    int num_simulations;
    int batch_size;
    float dirichlet_alpha;
    float dirichlet_epsilon;
    int sims_done = 0;

    // Pending leaves waiting for neural network evaluation
    std::vector<MCTSNode*> pending_leaves;

    MCTSSearch(const std::string& fen, float c_puct, int num_sims, int batch_size,
               float dir_alpha, float dir_epsilon)
        : c_puct(c_puct), num_simulations(num_sims), batch_size(batch_size),
          dirichlet_alpha(dir_alpha), dirichlet_epsilon(dir_epsilon) {
        root = std::make_unique<MCTSNode>();
        root->board.setFen(fen);
    }

    // Called once after root expansion to add Dirichlet noise
    void add_noise() {
        if (root->children.empty()) return;
        std::mt19937 rng(std::random_device{}());
        int n = root->children.size();
        std::vector<float> noise(n);
        std::gamma_distribution<float> gamma(dirichlet_alpha, 1.0f);
        float sum = 0;
        for (int i = 0; i < n; i++) { noise[i] = gamma(rng); sum += noise[i]; }
        for (int i = 0; i < n; i++) noise[i] /= sum;
        for (int i = 0; i < n; i++) {
            root->children[i]->prior =
                (1 - dirichlet_epsilon) * root->children[i]->prior +
                dirichlet_epsilon * noise[i];
        }
    }

    // Encode root board for initial evaluation
    py::array_t<float> encode_root() {
        auto result = py::array_t<float>({1, NUM_PLANES, 8, 8});
        encode_board(root->board, result.mutable_data());
        return result;
    }

    // Expand root with initial policy
    void expand_root(py::array_t<float> policy) {
        root->expand(policy.data());
        add_noise();
    }

    // Select a batch of leaves, return their encoded boards
    // Returns shape (N, 19, 8, 8) or empty if search is complete
    py::array_t<float> select_leaves() {
        pending_leaves.clear();
        std::vector<MCTSNode*> terminal_leaves;
        std::vector<float> terminal_values;

        int cur_batch = std::min(batch_size, num_simulations - sims_done);

        for (int i = 0; i < cur_batch; i++) {
            MCTSNode* leaf = root->select_leaf(c_puct);

            auto [reason, result] = leaf->board.isGameOver();
            if (result != GameResult::NONE) {
                float v = 0.0f;
                if (result == GameResult::WIN) v = 1.0f;
                else if (result == GameResult::LOSE) v = -1.0f;
                terminal_leaves.push_back(leaf);
                terminal_values.push_back(v);
            } else {
                leaf->add_virtual_loss();
                pending_leaves.push_back(leaf);
            }
        }

        // Process terminal nodes immediately
        for (size_t i = 0; i < terminal_leaves.size(); i++) {
            terminal_leaves[i]->backup(terminal_values[i]);
            sims_done++;
        }

        if (pending_leaves.empty()) {
            return py::array_t<float>({0, NUM_PLANES, 8, 8});
        }

        // Encode all pending leaf boards
        int n = pending_leaves.size();
        auto boards = py::array_t<float>({n, NUM_PLANES, 8, 8});
        float* data = boards.mutable_data();
        for (int i = 0; i < n; i++) {
            encode_board(pending_leaves[i]->board, data + i * NUM_PLANES * 64);
        }
        return boards;
    }

    // Process neural network results for pending leaves
    void process_results(py::array_t<float> policies, py::array_t<float> values) {
        const float* pol_data = policies.data();
        const float* val_data = values.data();

        for (size_t i = 0; i < pending_leaves.size(); i++) {
            MCTSNode* leaf = pending_leaves[i];
            leaf->revert_virtual_loss();
            leaf->expand(pol_data + i * ACTION_SPACE);
            leaf->backup(val_data[i]);
            sims_done++;
        }
        pending_leaves.clear();
    }

    bool is_complete() const { return sims_done >= num_simulations; }

    py::array_t<float> get_action_probs() {
        auto result = py::array_t<float>(ACTION_SPACE);
        float* data = result.mutable_data();
        std::memset(data, 0, ACTION_SPACE * sizeof(float));

        float total = 0;
        for (auto& child : root->children) {
            data[child->action] = child->visit_count;
            total += child->visit_count;
        }
        if (total > 0) {
            for (int i = 0; i < ACTION_SPACE; i++) data[i] /= total;
        }
        return result;
    }
};


PYBIND11_MODULE(mcts_cpp, m) {
    m.doc() = "C++ MCTS for AlphaZero chess";

    py::class_<MCTSSearch>(m, "MCTSSearch")
        .def(py::init<const std::string&, float, int, int, float, float>(),
             py::arg("fen"), py::arg("c_puct"), py::arg("num_sims"),
             py::arg("batch_size"), py::arg("dir_alpha"), py::arg("dir_epsilon"))
        .def("encode_root", &MCTSSearch::encode_root)
        .def("expand_root", &MCTSSearch::expand_root)
        .def("select_leaves", &MCTSSearch::select_leaves)
        .def("process_results", &MCTSSearch::process_results)
        .def("is_complete", &MCTSSearch::is_complete)
        .def("get_action_probs", &MCTSSearch::get_action_probs);
}
