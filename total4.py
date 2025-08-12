import pygame
import os
from bb_board import Board
from bb_sidebar import Sidebar

class Game:
    def __init__(self, board_size, perimeter, sidebar_width):
        """Initialize the game with board setup and state."""
        self.board = Board(board_size, perimeter)
        self.sidebar = Sidebar(board_size + 2 * perimeter, 0, sidebar_width, board_size + 2 * perimeter)  # Sidebar setup
        self.selected_piece = None
        self.move_text = ""  # Text for logging or display
        self.move_log = []  # Move history
        self.turn = "white"  # White starts the game
        self.is_check = False  # Check indicator
        self.depth = 2  # Initialize depth
        self.width = 5  # Initialize width
        self.predicted_moves = []  # Store predicted moves

    def predict(self, tree_width, tree_depth, method="top_scores"):
        """Selects a move prediction method."""
        if method == "top_scores":
            return self.predict_top_scores(tree_width, tree_depth)
        elif method == "minimax":
            return self.predict_minimax(tree_width, tree_depth)
        else:
            raise ValueError(f"Unknown prediction method: {method}")


    def predict_top_scores(self, tree_width, tree_depth):
        """Ranks moves by heatmap score, recursively simulating for depth levels."""
        if tree_depth == 0:
            return []  # Stop recursion at depth 0

        possible_moves = self.board.get_available_moves(self.turn)
        scored_moves = []

        for move in possible_moves:
            saved_state = self.board.get_bitboard_state()
            self.board.apply_move_bitboard(move)

            heatmap_score = self.board.calculate_global_heatmap()
            self.board.set_bitboard_state(saved_state)  # Restore state

            scored_moves.append((move, heatmap_score))

        # Sort moves by heatmap score (descending)
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        best_moves = scored_moves[:tree_width]  # Take top X moves

        # Recursively explore deeper levels
        best_moves_with_depth = []
        for move, score in best_moves:
            saved_state = self.board.get_bitboard_state()
            self.board.apply_move_bitboard(move)

            future_moves = self.predict_top_scores(tree_width, tree_depth - 1)  # Recurse with reduced depth
            best_moves_with_depth.append((move, score, future_moves))

            self.board.set_bitboard_state(saved_state)  # Restore state

        return best_moves_with_depth

    def predict_minimax(self, tree_width, tree_depth):
        """Minimax search with alpha-beta pruning."""
        # Minimax function (from previous message) can be added here when needed.
        """Minimax algorithm with alpha-beta pruning using heatmap scores."""
        if tree_depth == 0:
            return self.board.calculate_global_heatmap()  # Base case: return evaluation

        possible_moves = self.board.get_available_moves(self.turn)
        
        if maximizing:  # White's turn (maximize)
            max_eval = float("-inf")
            best_moves = []

            for move in possible_moves[:tree_width]:  # Prune by width
                saved_state = self.board.get_bitboard_state()
                self.board.apply_move_bitboard(move)
                self.turn = "black"  # Switch turn

                evaluation = self.predict(tree_width, tree_depth - 1, alpha, beta, maximizing=False)

                self.board.set_bitboard_state(saved_state)  # Undo move
                self.turn = "white"

                if evaluation > max_eval:
                    max_eval = evaluation
                    best_moves = [move]
                elif evaluation == max_eval:
                    best_moves.append(move)

                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    break  # Alpha-beta pruning

            return best_moves if tree_depth == self.depth else max_eval

        else:  # Black's turn (minimize)
            min_eval = float("inf")

            for move in possible_moves[:tree_width]:
                saved_state = self.board.get_bitboard_state()
                self.board.apply_move_bitboard(move)
                self.turn = "white"

                evaluation = self.predict(tree_width, tree_depth - 1, alpha, beta, maximizing=True)

                self.board.set_bitboard_state(saved_state)  # Undo move
                self.turn = "black"

                min_eval = min(min_eval, evaluation)
                beta = min(beta, evaluation)
                if beta <= alpha:
                    break  # Alpha-beta pruning

            return min_eval

    def select_piece(self, row, col):
        """Select a piece only if it matches the current player's color and exists at the given position."""
        piece = self.board.get_piece(row, col)
        if piece and piece.color == self.turn:
            self.selected_piece = piece
            self.available_moves = [move for move in self.board.legal_moves if move[0] == piece]
        else:
            # Clear selection if an invalid piece or empty square is clicked
            self.selected_piece = None
            self.available_moves = []
            
    def make_move(self, piece, new_row, new_col, win):
        """Executes a move and triggers updates for available moves and influence map."""
        old_row = piece.row
        old_col = piece.col
        
        if (new_row, new_col) in [(move[1], move[2]) for move in self.available_moves]:
            # Handle en passant
            if isinstance(piece, Pawn) and new_col != old_col and self.board.get_piece(new_row, new_col) is None:
                
                self.board.board[old_row][new_col] = None
            self.board.move_piece(piece, new_row, new_col)
            self.move_log.append((self.turn, piece, new_row, new_col))
            

            # Handle castling
            if isinstance(piece, King):
                if new_col == 2:
                    rook = self.board.get_piece(piece.row, 0)
                    self.board.move_piece(rook, piece.row, 3)
                if new_col == 6: 
                    rook = self.board.get_piece(piece.row, 7)
                    self.board.move_piece(rook, piece.row, 5)


    def get_square_under_mouse(self, mouse_pos):
        """Determine the board square (row, col) under the mouse position."""
        x, y = mouse_pos
        row = (y-self.board.perimeter) // self.board.square_size
        col = (x-self.board.perimeter) // self.board.square_size
        return row, col

    def update_game(self, move):
        """Handles all game updates after a move, passing necessary data to UI."""
        self.board.update(move)
        
        # Get new move predictions
        self.predicted_moves = self.predict(self.tree_width, self.tree_depth, method="top_scores")



    def draw(self, win):
        """Draws the game board, pieces, and overlays"""
        self.board.draw(win)  # Draw chess pieces
        if self.sidebar.show_heatmap:
            self.board.draw_heatmap(win)  # Heatmap overlay
        self.sidebar.draw(win)  # Sidebar UI

    def draw_heatmap(self, win):
        """Draw heatmap overlay"""
        if not self.sidebar.show_heatmap:
            return
        for r in range(8):
            for c in range(8):
                value = self.board.heatmap[r, c]
                if value == 0:
                    continue
                color = (255, 0, 0) if value > 0 else (0, 0, 255)
                pygame.draw.rect(win, color, (c * 100, r * 100, 100, 100), 2)

# Constants for screen size
BOARD_PERIMETER_WIDTH = 20
BOARD_SIZE = 800
SIDEBAR_WIDTH = 900
WINDOW_WIDTH = BOARD_SIZE + SIDEBAR_WIDTH + 2 * BOARD_PERIMETER_WIDTH
WINDOW_HEIGHT = BOARD_SIZE + 2 * BOARD_PERIMETER_WIDTH  # Adjust as needed

def main():
    pygame.init()
    win = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Chess Game")
    # Initialize game with specified board size
    game = Game(BOARD_SIZE, BOARD_PERIMETER_WIDTH,SIDEBAR_WIDTH)
    font = pygame.font.SysFont(None, 36)
    
    clock = pygame.time.Clock()
    run = True
    FPS = 60
    move_made = False  # Flag to determine when a new move is made
    while run:
        clock.tick(FPS)  # Limit frame rate

        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos()
                row, col = game.get_square_under_mouse(pos)
                if game.selected_piece:
                    if (row, col) in [(move[1], move[2]) for move in game.available_moves]:
                        game.make_move(game.selected_piece, row, col, win)
                        move_made = True
                        game.piece_selected = False
                    else:
                        game.select_piece(row, col)
                        game.piece_selected = bool(game.selected_piece)
                else:
                    game.select_piece(row, col)
                    game.piece_selected = bool(game.selected_piece)

            # Handle Sidebar Events
            game.sidebar.handle_event(event)

        # Clear Window
        win.fill((0, 0, 0))

        # Draw Game Board and Pieces
        game.board.draw(win, game.selected_piece)

        # Recalculate heatmap, control, and available moves after a move
        if move_made:
            game.update_game_state(game.sidebar.tree_width, game.sidebar.tree_depth)
            game.update_sidebar(win, game.predicted_moves)
            move_made = False  

        # **Draw Heatmap if Enabled**
        if game.sidebar.show_heatmap:
            game.board.draw_heatmap(win)  

        # Draw Sidebar
        game.sidebar.draw(win, game.turn, game.board.predictions)

        pygame.display.update()


if __name__ == "__main__":
    main()



import pygame
import pandas as pd
import os
import numpy as np

class Board:
    def __init__(self, board_size, perimeter):
        self.board_size = board_size
        self.square_size = (board_size - 2 * perimeter) // 8  # Calculate the size of each square
        self.perimeter = perimeter 
        self.bitboards = self.init_bitboards()  # Store board state in bitboards
        self.heatmap = np.zeros((8, 8), dtype=int)  # Heatmap for influence
        self.fen_log = []  # Stores FEN for debugging & history
        self.predictions = []  # Store evaluated moves
        self.turn = "white"

        # Precomputed bitboard values
        self.white_pieces = (self.bitboards["white_pawns"] | self.bitboards["white_knights"] | 
                            self.bitboards["white_bishops"] | self.bitboards["white_rooks"] | 
                            self.bitboards["white_queen"] | self.bitboards["white_king"])

        self.black_pieces = (self.bitboards["black_pawns"] | self.bitboards["black_knights"] | 
                            self.bitboards["black_bishops"] | self.bitboards["black_rooks"] | 
                            self.bitboards["black_queen"] | self.bitboards["black_king"])

        self.occupied = self.white_pieces | self.black_pieces
        self.empty = ~self.occupied & 0xFFFFFFFFFFFFFFFF  # Ensure only 64 bits are considered

        # Castling & En Passant
        self.castling_rights = {"white": {"K": True, "Q": True}, "black": {"k": True, "q": True}}
        self.en_passant_target = None  # Square where en passant is possible

    def init_bitboards(self):
        """Initialize all bitboards for piece positions and metadata"""
        return {
            "white_pawns": 0x000000000000FF00,
            "black_pawns": 0x00FF000000000000,
            "white_knights": 0x0000000000000042,
            "black_knights": 0x4200000000000000,
            "white_bishops": 0x0000000000000024,
            "black_bishops": 0x2400000000000000,
            "white_rooks": 0x0000000000000081,
            "black_rooks": 0x8100000000000000,
            "white_queen": 0x0000000000000008,
            "black_queen": 0x0800000000000000,
            "white_king": 0x0000000000000010,
            "black_king": 0x1000000000000000,
        }

    def get_bitboard_state(self):
        """Returns a snapshot of the current board state."""
        return (
            self.white_pawns, self.black_pawns, self.white_knights, self.black_knights,
            self.white_bishops, self.black_bishops, self.white_rooks, self.black_rooks,
            self.white_queen, self.black_queen, self.white_king, self.black_king,
            self.black_pieces, self.white_pieces, self.occupied, self.empty
        )

    def set_bitboard_state(self, state):
        """Restores a saved board state."""
        (
            self.white_pawns, self.black_pawns, self.white_knights, self.black_knights,
            self.white_bishops, self.black_bishops, self.white_rooks, self.black_rooks,
            self.white_queen, self.black_queen, self.white_king, self.black_king,
            self.black_pieces, self.white_pieces, self.occupied, self.empty
        ) = state


    def calculate_global_heatmap(self):
        """Computes a score by summing influence of both colors."""
        white_control = self.calculate_control("white")
        black_control = self.calculate_control("black")
        return sum(white_control) - sum(black_control)

    def get_available_moves(board_state, color):
        """Returns available moves and influence for all pieces."""
        occupied = board_state['occupied']
        empty = ~occupied

        # Initialize dictionaries
        moves = {}
        influence = {}

        if color == "white":
            pawns = board_state['white_pawns']
            knights = board_state['white_knights']
            king = board_state['white_king']
            rooks = board_state['white_rooks']
            bishops = board_state['white_bishops']
            queens = board_state['white_queens']
            enemy_occupied = board_state['black_pieces']
        else:
            pawns = board_state['black_pawns']
            knights = board_state['black_knights']
            king = board_state['black_king']
            rooks = board_state['black_rooks']
            bishops = board_state['black_bishops']
            queens = board_state['black_queens']
            enemy_occupied = board_state['white_pieces']

        moves['pawn'], influence['pawn'] = pawn_moves(pawns, occupied, empty, white=(color=="white"))
        moves['knight'], influence['knight'] = knight_moves(knights)
        moves['king'], influence['king'] = king_moves(king)
        moves['rook'], influence['rook'] = rook_moves(rooks, occupied)
        moves['bishop'], influence['bishop'] = bishop_moves(bishops, occupied)
        moves['queen'], influence['queen'] = queen_moves(queens, occupied)

        return moves, influence

    def get_legal_moves(board_state, color):
        """Filters out illegal moves that leave the king in check."""
        moves, influence = get_available_moves(board_state, color)
        legal_moves = {}

        king_position = board_state['white_king'] if color == "white" else board_state['black_king']
        enemy_color = "black" if color == "white" else "white"

        for piece, move_set in moves.items():
            legal_moves[piece] = 0  # Start with no legal moves

            while move_set:
                move = move_set & -move_set  # Get the least significant 1-bit move
                move_set ^= move  # Remove this move from the set

                # Simulate the move
                new_board = apply_move_bitboard(board_state, (piece, move))

                # Check if king is in check
                _, enemy_influence = get_available_moves(new_board, enemy_color)
                if not (enemy_influence['king'] & king_position):
                    legal_moves[piece] |= move  # Add move if legal

        return legal_moves

    def apply_move_bitboard(self, move):
        """Applies a move using bitboard manipulation."""
        start_square, end_square, piece = move

        # Convert move positions to bitboard representation
        start_mask = 1 << start_square
        end_mask = 1 << end_square

        # Find which bitboard the piece belongs to
        for key in self.bitboards:
            if self.bitboards[key] & start_mask:  # Found piece
                self.bitboards[key] ^= start_mask  # Remove from start
                self.bitboards[key] |= end_mask  # Move to destination
                break

        # Handle captures: Remove opponent piece from bitboard
        for key in self.bitboards:
            if self.bitboards[key] & end_mask and key.startswith(self.opponent_color()):
                self.bitboards[key] ^= end_mask  # Remove captured piece

        # Update metadata
        self.update_metadata()

#Drawing 

    def draw(self, win, selected_piece):
        """Draws the chessboard and pieces."""
        colors = [(232, 235, 239), (125, 135, 150)]  # White and black colors for the squares
        # Draw the board
        for row in range(8):
            for col in range(8):
                color = colors[(row + col) % 2]
                pygame.draw.rect(win, color, pygame.Rect(
                    col * self.square_size, row * self.square_size,
                    self.square_size, self.square_size
                ))

        # Draw pieces (assuming we have a function to get piece positions)
        for piece, bitboard in self.bitboards.items():
            self.draw_pieces(win, bitboard, piece)
            
        # Highlight the selected piece
        if selected_piece:
            x = selected_piece.col * self.square_size + self.perimeter
            y = selected_piece.row * self.square_size + self.perimeter
            pygame.draw.rect(win, (0, 0, 0), (x, y, self.square_size, self.square_size), 5)  # Thick black border
        
    def draw_pieces(self, win, bitboard, piece):
        """Draws a piece on the board using bitboard positions."""
        piece_images = {
            "white_pawns": pygame.image.load("assets/wp.png"),
            "black_pawns": pygame.image.load("assets/bp.png"),
            "white_knights": pygame.image.load("assets/wn.png"),
            "black_knights": pygame.image.load("assets/bn.png"),
            "white_bishops": pygame.image.load("assets/wb.png"),
            "black_bishops": pygame.image.load("assets/bb.png"),
            "white_rooks": pygame.image.load("assets/wr.png"),
            "black_rooks": pygame.image.load("assets/br.png"),
            "white_queen": pygame.image.load("assets/wq.png"),
            "black_queen": pygame.image.load("assets/bq.png"),
            "white_king": pygame.image.load("assets/wk.png"),
            "black_king": pygame.image.load("assets/bk.png"),
            
        }
        
        if piece in piece_images:
            img = pygame.transform.scale(piece_images[piece], (self.square_size, self.square_size))

            while bitboard:
                square = (bitboard & -bitboard).bit_length() - 1
                row, col = divmod(square, 8)
                win.blit(img, (col * self.square_size, (7 - row) * self.square_size))
                bitboard &= bitboard - 1  # Remove the LSB (processed piece)

#Board Update
    def update(self, move):
        """Handles a move and updates available moves."""
        # Apply the move
        self.apply_move_bitboard(move)
        # Switch turn
        self.current_turn = "black" if self.current_turn == "white" else "white"

        # Recompute available moves for the new player
        self.available_moves, self.influence = get_available_moves(self.board_state, self.current_turn)

        self.update_heatmap()
        
    def update_heatmap(self):
        """Updates heatmap based on piece influence."""
        self.heatmap.fill(0)  # Reset heatmap

        for color in ["white", "black"]:
            attacks = self.get_all_attacks(color)  # Get influence from all pieces
            value = 1 if color == "white" else -1

            while attacks:
                square = attacks.bit_length() - 1
                rank, file = divmod(square, 8)
                self.heatmap[7 - rank, file] += value
                attacks &= attacks - 1  # Remove LSB


#Pieces moves & influence

    def get_piece(self, row, col):
        """Return the piece at a given (row, col), or None if empty."""
        square = (7 - row) * 8 + col  # Adjust for flipped Y
        mask = 1 << square
        for piece, bitboard in self.bitboards.items():
            if bitboard & mask:
                return {"type": piece, "square": square, "row": row, "col": col}
        return None


    def pawn_moves(pawns, occupied, empty, white=True):
        """Computes legal pawn moves and influence."""
        if white:
            single_push = (pawns << 8) & empty
            double_push = ((single_push & 0x0000000000FF0000) << 8) & empty
            left_capture = (pawns << 7) & occupied & ~0x8080808080808080
            right_capture = (pawns << 9) & occupied & ~0x0101010101010101
        else:
            single_push = (pawns >> 8) & empty
            double_push = ((single_push & 0x0000FF0000000000) >> 8) & empty
            left_capture = (pawns >> 9) & occupied & ~0x8080808080808080
            right_capture = (pawns >> 7) & occupied & ~0x0101010101010101

        moves = single_push | double_push | left_capture | right_capture
        influence = left_capture | right_capture  # Pawn attacks only diagonally

        return moves, influence

    def knight_moves(knights):
        """Computes knight moves and influence."""
        L1 = (knights & ~0x0101010101010101) >> 1
        L2 = (knights & ~0x0303030303030303) >> 2
        R1 = (knights & ~0x8080808080808080) << 1
        R2 = (knights & ~0xC0C0C0C0C0C0C0C0) << 2
        moves = (
            (L1 << 16) | (L1 >> 16) |
            (L2 << 8)  | (L2 >> 8)  |
            (R1 << 16) | (R1 >> 16) |
            (R2 << 8)  | (R2 >> 8)
        )
        return moves, moves  # Influence is same as moves

    def king_moves(king):
        """Computes king moves and influence."""
        moves = (
            (king << 8) | (king >> 8) |  
            ((king & ~0x8080808080808080) << 1) | ((king & ~0x0101010101010101) >> 1) |  
            ((king & ~0x8080808080808080) << 9) | ((king & ~0x8080808080808080) >> 7) |  
            ((king & ~0x0101010101010101) << 7) | ((king & ~0x0101010101010101) >> 9)
        )
        return moves, moves

    def rook_moves(rook, occupied):
        """Computes rook moves and influence using sliding attack."""
        def sliding_attacks(direction, board, blockers):
            moves = 0
            while board:
                board = (board << direction) if direction > 0 else (board >> -direction)
                moves |= board
                if board & blockers:
                    break
            return moves

        moves = (
            sliding_attacks(8, rook, occupied) |  # Up
            sliding_attacks(-8, rook, occupied) | # Down
            sliding_attacks(1, rook & ~0x8080808080808080, occupied) |  # Right
            sliding_attacks(-1, rook & ~0x0101010101010101, occupied)   # Left
        )
        return moves, moves

    def bishop_moves(bishop, occupied):
        """Computes bishop moves and influence using sliding attack."""
        def sliding_attacks(direction, board, blockers):
            moves = 0
            while board:
                board = (board << direction) if direction > 0 else (board >> -direction)
                moves |= board
                if board & blockers:
                    break
            return moves

        moves = (
            sliding_attacks(9, bishop & ~0x8080808080808080, occupied) |  # Up-right
            sliding_attacks(7, bishop & ~0x0101010101010101, occupied) |  # Up-left
            sliding_attacks(-7, bishop & ~0x8080808080808080, occupied) | # Down-right
            sliding_attacks(-9, bishop & ~0x0101010101010101, occupied)   # Down-left
        )
        return moves, moves

    def queen_moves(queen, occupied):
        """Computes queen moves and influence (rook + bishop)."""
        r_moves, r_influence = rook_moves(queen, occupied)
        b_moves, b_influence = bishop_moves(queen, occupied)
        return r_moves | b_moves, r_influence | b_influence

    def can_castle(self, color, side):
        """Returns True if castling is legal for given color and side ('k' or 'q')."""
        if color == 'w':
            king_start, rook_start = 0b00010000, (0b10000000 if side == 'q' else 0b00000001)  # e1, a1/h1
            empty_squares = 0b01100000 if side == 'k' else 0b00001110  # f1-g1 / b1-d1
            check_squares = 0b00100000 if side == 'k' else 0b00000100  # King moves f1/g1 / d1

        else:
            king_start, rook_start = 0b00010000 << 56, (0b10000000 << 56 if side == 'q' else 0b00000001 << 56)  # e8, a8/h8
            empty_squares = 0b01100000 << 56 if side == 'k' else 0b00001110 << 56  # f8-g8 / b8-d8
            check_squares = 0b00100000 << 56 if side == 'k' else 0b00000100 << 56  # f8/g8 / d8

        # Verify castling rights
        if ('K' not in self.castling_rights if color == 'w' else 'k' not in self.castling_rights):
            return False

        # Check if pieces are in the way
        if self.occupied & empty_squares:
            return False

        # Ensure no attacks on king path
        if self.is_king_in_check(color):
            return False

        opponent_attack_bb = self.get_attack_bitboard('b' if color == 'w' else 'w')
        if opponent_attack_bb & check_squares:
            return False

        return True


#Visualization for humans
    def print_board_from_bitboards(board):
        """Prints a full chessboard from bitboards"""
        display = [["."] * 8 for _ in range(8)]

        # Loop through each piece bitboard
        for piece, bb in board.items():
            while bb:
                square = (bb & -bb).bit_length() - 1  # Get the lowest set bit (LSB)
                rank, file = divmod(square, 8)
                display[7 - rank][file] = PIECE_SYMBOLS.get(piece, "?")  # Convert to symbol
                bb &= bb - 1  # Remove LSB

        # Print the board
        for row in display:
            print(" ".join(row))
        print("\n")

    def to_fen(self):
        """Converts the current board state into a FEN string."""
        board_fen = ""
        
        # Convert bitboards into piece placement (rows from 8th rank to 1st)
        for rank in range(7, -1, -1):
            empty_squares = 0
            for file in range(8):
                square = rank * 8 + file
                piece_found = False
                for piece, bb in self.bitboards.items():
                    if bb & (1 << square):
                        if empty_squares:
                            board_fen += str(empty_squares)
                            empty_squares = 0
                        board_fen += piece
                        piece_found = True
                        break
                if not piece_found:
                    empty_squares += 1
            if empty_squares:
                board_fen += str(empty_squares)
            board_fen += "/"
        
        board_fen = board_fen[:-1]  # Remove last "/"

        # Active color
        active_color = "w" if self.white_to_move else "b"

        # Castling rights
        castling_fen = "".join(k for k, v in self.castling_rights.items() if v) or "-"

        # En passant target
        en_passant_fen = "-" if self.en_passant_target == -1 else self.square_to_coord(self.en_passant_target)

        # Full FEN string
        return f"{board_fen} {active_color} {castling_fen} {en_passant_fen} 0 1"

    def from_fen(self, fen):
        """Loads a board state from a FEN string."""
        parts = fen.split()
        
        # 1. Load piece positions
        self.bitboards = {p: 0 for p in "PNBRQKpnbrqk"}
        ranks = parts[0].split("/")
        for rank in range(8):
            file = 0
            for char in ranks[7 - rank]:  # Read from rank 8 down to rank 1
                if char.isdigit():
                    file += int(char)
                else:
                    square = rank * 8 + file
                    self.bitboards[char] |= (1 << square)
                    file += 1

        # 2. Active color
        self.white_to_move = (parts[1] == "w")

        # 3. Castling rights
        self.castling_rights = {"K": "K" in parts[2], "Q": "Q" in parts[2],
                                "k": "k" in parts[2], "q": "q" in parts[2]}

        # 4. En passant target
        self.en_passant_target = -1 if parts[3] == "-" else self.coord_to_square(parts[3])


import pygame
import sys

class Sidebar:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height        
        self.rect = pygame.Rect(x, y, width, height)
        
        # Fonts & Colors
        self.font = pygame.font.SysFont('Arial', 18, bold=True)
        self.small_font = pygame.font.SysFont('Arial', 14)
        self.bg_color = (50, 50, 50)
        self.text_color = (255, 255, 255)
        self.highlight_color = (120, 220, 120)
        
        # UI Elements
        self.input_active = False
        self.input_text = ''
        self.input_box = pygame.Rect(x + 10, y + height - 60, width - 20, 35)
        
        # Sections
        self.title_area = pygame.Rect(x + 10, y + 10, width - 20, 30)
        self.moves_area = pygame.Rect(x + 10, y + 50, width - 20, 100)
        self.method_area = pygame.Rect(x + 10, y + 170, width - 20, 90)
        self.input_area = pygame.Rect(x + 10, y + height - 70, width - 20, 40)
        
        # Methodology Selection
        self.methods = ['Heat', 'Belt', 'Integrity']
        self.selected_method = 0
        # Heatmap Toggle Button
        self.show_heatmap = False
        self.heatmap_button = pygame.Rect(self.method_area.x, self.method_area.y + 80, 20, 20)

        # Move Recommendations
        self.best_moves = []
    
    def draw(self, win, turn, predictions):
        pygame.draw.rect(win, self.bg_color, self.rect)
        
        # Display current turn
        turn_surf = self.font.render(f"Turn: {turn}", True, self.text_color)
        win.blit(turn_surf, (self.title_area.x, self.title_area.y + 30))

        # Display Move Predictions
        self.best_moves = predictions  # Store predictions
        for i, move in enumerate(self.best_moves[:5]):  # Show top 5
            move_surf = self.small_font.render(
                f"{move[0]} -> {move[1]} ({move[3]:.1f})", True, self.text_color
            )
            win.blit(move_surf, (self.moves_area.x, self.moves_area.y + i * 20))
        
        # Method Selection
        for i, method in enumerate(self.methods):
            color = self.highlight_color if i == self.selected_method else self.text_color
            method_surf = self.small_font.render(method, True, color)
            win.blit(method_surf, (self.method_area.x, self.method_area.y + i * 25))
        
        # Draw Heatmap Toggle
        pygame.draw.rect(win, (200, 200, 200), self.heatmap_button, 2)
        if self.show_heatmap:
            pygame.draw.line(win, (200, 200, 200), self.heatmap_button.topleft, self.heatmap_button.bottomright, 2)
            pygame.draw.line(win, (200, 200, 200), self.heatmap_button.topright, self.heatmap_button.bottomleft, 2)

        heatmap_label = self.small_font.render("Show Heatmap", True, self.text_color)
        win.blit(heatmap_label, (self.heatmap_button.right + 5, self.heatmap_button.y - 2))

                
        # Input Box
        pygame.draw.rect(win, (200, 200, 200), self.input_box, 2)
        input_surf = self.small_font.render(self.input_text, True, self.text_color)
        win.blit(input_surf, (self.input_box.x + 5, self.input_box.y + 5))
    
    def update_sidebar(self, win, predictions):
        """Updates sidebar UI based on given predictions (no Game calls)."""
        self.best_moves = predictions  # Store new predictions

        # Redraw UI
        pygame.draw.rect(win, self.bg_color, self.rect)  # Sidebar background
        self.draw(win)  # Call only its own draw method
        pygame.display.update()


    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.input_box.collidepoint(event.pos):
                self.input_active = not self.input_active
            else:
                self.input_active = False
            if self.heatmap_button.collidepoint(event.pos):
                self.show_heatmap = not self.show_heatmap
        elif event.type == pygame.KEYDOWN and self.input_active:
            if event.key == pygame.K_RETURN:
                self.input_text = ''
            elif event.key == pygame.K_BACKSPACE:
                self.input_text = self.input_text[:-1]
            else:
                self.input_text += event.unicode
