// Backtracking as an example to solve the "Four Connect" game problem.
// "Three Connect" is a faster variant of the classical "Four Connect" game
// (i.e. same rules, but only three connected pieces are needed to win).
//
// The game is played on a NxM grid where two players take turns placing their markers (1 and 2).
// The objective is to connect four of one's own markers in a row, either horizontally, vertically, or diagonally.
// The original "Four Connect" game setup (4 connected pieces win on a 7x6 grid) is too timeconsuming without further optimization.
// To study backtracking, a faster "Three Connect" variant can be used (e.g. 3 connected pieces win on a 4x3 grid).
// The program uses backtracking to explore all possible game states and determine if there is a winning strategy for the first player.
//
// Example output:
// 
// 2-Connect game on a 4x4 board:
// Number of winning games for player 1: 4
// Number of winning games for player 2: 0
// Number of remis games:                0
// Backtracking winner using best answers: 1 (i.e. player 1 wins if both play optimal)
// Creation and backtracking executed in 0.000 sec.
// 
// 3-Connect game on a 3x4 board:
// Number of winning games for player 1: 170
// Number of winning games for player 2: 183
// Number of remis games:                85
// Backtracking winner using best answers: 0 (i.e. remis if both play optimal)
// Creation and backtracking executed in 0.000 sec.
// 
// 3-Connect game on a 4x3 board:
// Number of winning games for player 1: 33
// Number of winning games for player 2: 13
// Number of remis games:                0
// Backtracking winner using best answers: 1 (i.e. player 1 wins if both play optimal)
// Creation and backtracking executed in 0.000 sec.
// 
// 3-Connect game on a 4x4 board:
// Number of winning games for player 1: 220
// Number of winning games for player 2: 186
// Number of remis games:                3
// Backtracking winner using best answers: 1 (i.e. player 1 wins if both play optimal)
// Creation and backtracking executed in 0.000 sec.
// 
// 4-Connect game on a 4x4 board:
// Number of winning games for player 1: 1423077
// Number of winning games for player 2: 1239050
// Number of remis games:                9635303
// Backtracking winner using best answers: 0 (i.e. remis if both play optimal)
// Creation and backtracking executed in 1.522 sec.
// 
// 4-Connect game on a 5x4 board:
// Number of winning games for player 1: 219786333
// Number of winning games for player 2: 190707951
// Number of remis games:                1308518933
// Backtracking winner using best answers: 0 (i.e. remis if both play optimal)
// Creation and backtracking executed in 201.554 sec.
//
// Without further optimizations, the classical "Four Connect" game on a 7x6 board is too time-consuming.

#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

/*
** Class to represent the NxM board for the Four Connect game
*/
template<size_t COLS, size_t ROWS, size_t NUM_PIECES_TO_WIN> class Board
{
public:

    Board(std::array<std::array<int8_t, COLS>, ROWS>* board = 0) : m_board(board), m_winner(-1)
    {
        if (!m_board)
        {
            m_board = new std::array<std::array<int8_t, COLS>, ROWS>();
            for (auto& row : (*m_board))
                row.fill(0);
        }
    }
    
    int& winner() { return m_winner; }
    const int& winner() const { return m_winner; }
    std::array<std::array<int8_t, COLS>, ROWS>* board() { return m_board; }
    inline void set_piece(int col, int row, int piece) { (*m_board)[row][col] = piece; }
    bool is_col_full(int col) const { return (*m_board)[0][col] != 0; }
   
    bool is_board_full() const
    {
        for (int col = 0; col < COLS; col++)
        {
            if (!is_col_full(col))
                return false;
        }
        return true;
    }

    int drop_piece(int col, int piece, int verbosity = 1)
    {
        // Drop a piece in the specified column (if it's not full yet) and check if there's a winner. Returns the row of the dropped piece (or -1 if the column is full).
        for (int row = ROWS - 1; row >= 0; row--)
        {
            if ((*m_board)[row][col] == 0)
            {
                (*m_board)[row][col] = piece;
                if (check_winner_at_drop(piece, col, row))
				{
                    m_winner = piece;
                    print(verbosity);
				}
                if (m_winner < 0 && row == 0 && is_board_full())
                {
                    m_winner = 0; // game completed with remis, if all columns are full
                    print(verbosity);
                }
                return row;
            }
        }
        return -1;
    }

    bool check_winner_at_drop(int piece, int col, int row) const
    {
        // Check from (col,row) vertically downward for connected pieces
        if ((1 + count_connected_pieces(piece, col, row + 1, 0, 1)) >= NUM_PIECES_TO_WIN)
            return true;
        // Check from (col,row) horizontally to the left and right for connected pieces
        if ((1 + count_connected_pieces(piece, col - 1, row, -1, 0) + count_connected_pieces(piece, col + 1, row, 1, 0)) >= NUM_PIECES_TO_WIN)
            return true;
        // Check from (col,row) diagonally (top left to bottom right) for connected pieces
        if ((1 + count_connected_pieces(piece, col - 1, row - 1, -1, -1) + count_connected_pieces(piece, col + 1, row + 1, 1, 1)) >= NUM_PIECES_TO_WIN)
            return true;
        // Check from (col,row) diagonally (top right to bottom left) for connected pieces
        if ((1 + count_connected_pieces(piece, col - 1, row + 1, -1, 1) + count_connected_pieces(piece, col + 1, row - 1, 1, -1)) >= NUM_PIECES_TO_WIN)
            return true;
        return false;
	}
    
    int count_connected_pieces(int piece, int col, int row, int delta_col, int delta_row) const
    {
        // Count the number of connected pieces of the same type in a given direction
        int num_connected = 0;
        int curr_col = col;
        int curr_row = row;
        while (0 <= curr_col && curr_col < COLS && 0 <= curr_row && curr_row < ROWS && (*m_board)[curr_row][curr_col] == piece)
        {
            num_connected += 1;
            curr_col += delta_col;
            curr_row += delta_row;
        }
        return num_connected;
    }
    
    void print(int verbosity = 1) const
    {
        if (verbosity >= 1)
        {
            std::cout << "board winner: " << m_winner << std::endl;
            for (const auto& row : *m_board)
            {
                for (const auto& cell : row)
                {
                    std::cout << static_cast<int>(cell) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
		}
    }

protected:

    std::array<std::array<int8_t, COLS>, ROWS>* m_board; // playing field, NxM grid, each cell has content 0 = empty, 1 = player 1 or 2 = player 2
	int m_winner; // if game is completed: 0 = remis, 1 = player 1 wins, 2 = player 2 wins, otherwise -1
};

/*
** Interface class to represent the Four Connect game
*/
class PlayConnectIF
{
public:
    virtual void generate_children() = 0;
    virtual int cols() const = 0;
    virtual int rows() const = 0;
    virtual int pieces_to_win() const = 0;
    virtual int backtracking_winner() const = 0;
    virtual size_t win_stats(int player) const = 0;
};

/*
** Class to play the Four Connect game on a given board
*/
template<size_t COLS, size_t ROWS, size_t NUM_PIECES_TO_WIN> class PlayConnect : public PlayConnectIF
{
public:

    PlayConnect(Board<COLS, ROWS, NUM_PIECES_TO_WIN>* board = 0, int player = 1, int verbosity = 1, std::array<size_t, 3>* win_stats = 0) : m_board(board?board->board():0), m_player(player), m_verbosity(verbosity), m_win_stats(win_stats)
    {
        if (!m_win_stats)
			m_win_stats = new std::array<size_t, 3>({ 0,0,0 });
	}

    void generate_children() override
    {
        struct PieceLocation
        {
            int col;
            int row;
		};
        // Generate the game tree recursively and backtrack to find a winning strategy for player 1
        std::vector<PlayConnect> child_nodes;
        std::array<PieceLocation, COLS> child_drops;
        child_nodes.reserve(COLS);
        int best_child_winner = (m_player == 1) ? 2 : 1; // worst case assumption: opponent wins, player loses
        std::array<int, COLS> col_search_order = column_search_order(); // start search in center column (columns near the center are more likely to win)
        for (int col_idx = 0, child_cnt = 0; col_idx < COLS; col_idx++)
        {
			int col = col_search_order[col_idx];
            if (!m_board.is_col_full(col))
            {
                PlayConnect child_node(&m_board, (m_player == 1) ? 2 : 1, m_verbosity, m_win_stats);
                int row = child_node.m_board.drop_piece(col, m_player, m_verbosity); // Drop a piece and check for winners
                if (child_node.m_board.winner() >= 0) // game completed, update statistics
                    (*m_win_stats)[child_node.m_board.winner()] += 1; // update win statistics
				child_node.m_board.set_piece(col, row, 0); // undo drop_piece for backtracking
                if (child_node.m_board.winner() == m_player) // If one of the piece drops led to an immediate win, there's no need to explore further
                {
                    best_child_winner = m_player;
                    break;
                }
                child_nodes.push_back(child_node);
                child_drops[child_cnt].col = col;
                child_drops[child_cnt].row = row;
				child_cnt++;
            }
        }
		// Expand all unfinished child nodes
        for (size_t child_cnt = 0; child_cnt < child_nodes.size() && best_child_winner != m_player; child_cnt++)
        { 
            if (child_nodes[child_cnt].m_board.winner() < 0) // game not completed, continue generating children
            {
                int col = child_drops[child_cnt].col, row = child_drops[child_cnt].row;
                child_nodes[child_cnt].m_board.set_piece(col, row, m_player); // redo drop_piece for backtracking
                child_nodes[child_cnt].generate_children();
                child_nodes[child_cnt].m_board.set_piece(col, row, 0); // undo drop_piece for backtracking
                if (child_nodes[child_cnt].m_board.winner() == m_player) // If one of the piece drops led to an immediate win, there's no need to explore further
                    best_child_winner = m_player;
            }
        }
        // backtrack: assign all generated child nodes at once for better performance
        for(size_t child_cnt = 0; child_cnt < child_nodes.size() && best_child_winner != m_player; child_cnt++)
        {
            if (child_nodes[child_cnt].m_board.winner() == m_player) // winning is always the very best choice
                best_child_winner = child_nodes[child_cnt].m_board.winner();
            else if (child_nodes[child_cnt].m_board.winner() == 0) // remis is always the second best choice
                best_child_winner = 0;
        }
        m_board.winner() = best_child_winner;
	}

    int cols() const override { return COLS; }
    int rows() const override { return ROWS; }
    int pieces_to_win() const override { return NUM_PIECES_TO_WIN; }
    int backtracking_winner() const override { return m_board.winner();       }
    size_t win_stats(int player) const override { return (*m_win_stats)[player]; }

protected:
    std::array<int, COLS> column_search_order() // Return the order in which columns are to be searched (center columns first for better performance)
    {
        std::array<int, COLS> order;
        int center = COLS / 2;
        order[0] = center;
        for (int i = 1; i <= center; i++)
        {
            if (center - i >= 0)
                order[2 * i - 1] = center - i;
            if (center + i < COLS)
                order[2 * i] = center + i;
        }
		return order;
    }
    Board<COLS, ROWS, NUM_PIECES_TO_WIN> m_board; // current state of the board
    int m_player;                    // current player (1 or 2), player 1 starts by default
    int m_verbosity;                 // verbosity level for printing
	std::array<size_t, 3>* m_win_stats; // statistics of wins: [number of remis, number of player 1 wins, number of player 2 wins]
};

int main() 
{
    // Create the initial boards for different game variants
	int verbosity = 0; // set verbosity to 0 and suppress board printing to speed up execution
#   ifdef _DEBUG
    verbosity = 1;
#   endif
    std::vector<PlayConnectIF*> games = {
        new PlayConnect<4, 4, 2>(0, 1, verbosity, 0), // cross-check: player 1 can always win on any board in the "two connect" variant
		new PlayConnect<3, 4, 3>(0, 1, verbosity, 0), // a fast "three connect" variant on a 3x4 board, player 1 starts and plays remis if both players play optimally
        new PlayConnect<4, 3, 3>(0, 1, 0, 0),         // a fast "three connect" variant on a 4x3 board, player 1 starts and wins if both players play optimally
        new PlayConnect<4, 4, 3>(0, 1, 0, 0),         // a fast "three connect" variant on a 4x4 board, player 1 starts and wins if both players play optimally
        new PlayConnect<4, 4, 4>(0, 1, 0, 0),         // "four connect" on a 4x4 board, player 1 starts, player 1 starts and plays remis if both players play optimally
		new PlayConnect<5, 4, 4>(0, 1, 0, 0),         // "four connect" on a 5x4 board, player 1 starts, player 1 starts and plays remis if both players play optimally
		new PlayConnect<7, 6, 4>(0, 1, 0, 0),         // "four connect" on the commonly used 7x6 board, player 1 starts, too time-consuming without further optimization
    };

	std::vector<std::string> winner_to_str = { "remis", "player 1 wins", "player 2 wins" };
	for (auto game : games)
    {
        std::cout << game->pieces_to_win() << "-Connect game on a " << game->cols() << "x" << game->rows() << " board:" << std::endl;
        std::chrono::time_point<std::chrono::system_clock> game_start_time = std::chrono::system_clock::now();
        game->generate_children(); // generate the game tree recursively and backtrack to find a winning strategy for player 1
        double seconds = (1.0e-6) * (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - game_start_time)).count();
        std::cout << "Number of winning games for player 1: " << game->win_stats(1) << std::endl;
        std::cout << "Number of winning games for player 2: " << game->win_stats(2) << std::endl;
        std::cout << "Number of remis games:                " << game->win_stats(0) << std::endl;
        std::cout << "Backtracking winner using best answers: " << game->backtracking_winner() << " (i.e. " << winner_to_str[game->backtracking_winner()] << " if both play optimal)" << std::endl;
        std::cout << "Creation and backtracking executed in " << std::fixed << std::setprecision(3) << seconds << " sec." << std::endl << std::endl;
    }
    return 0;
}
