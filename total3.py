import pygame
import os
from bb_board import Board
from bb_sidebar import Sidebar

class Game:
    def __init__(self, board_size, perimeter, sidebar_width):
        """Initialize the game with board setup and state."""
        self.board = Board(board_size,perimeter)
        self.sidebar = Sidebar(board_size + 2 * perimeter, 0, sidebar_width, board_size + 2 * perimeter)  # Sidebar setup
        self.board.set_initial_position()
        self.selected_piece = None
        self.move_text = ""  # Text for logging or display
        self.move_log = []  # Move history
        self.turn = "white"  # White starts the game
        self.is_check = False  # Check indicator
        self.depth = 2  # Initialize depth
        self.width = 5  # Initialize width

    def predict(self, depth=2, width=5):
        """
        Predicts the best move by simulating moves up to a certain depth.
        Uses heatmap balance as the scoring parameter.
        
        :param depth: Number of plies to look ahead.
        :param width: Max number of best moves to keep at each level.
        :return: List of best moves sorted by evaluation.
        """
        best_moves = []
        
        def minimax(board, depth, maximizing_player):
            """Recursive search function using minimax with bitboard optimizations."""
            if depth == 0:
                return board.calculate_global_score()  # Evaluate the position
            
            moves = board.get_legal_moves()
            if not moves:
                return -float("inf") if maximizing_player else float("inf")  # Avoids infinite loop

            move_scores = []
            
            for move in moves:
                board_state = board.save_state()  # Save current board state
                board.make_move(move)  # Apply move
                score = minimax(board, depth - 1, not maximizing_player)  # Recur with depth -1
                move_scores.append((move, score))
                board.restore_state(board_state)  # Undo move
            
            move_scores.sort(key=lambda x: x[1], reverse=maximizing_player)  # Sort moves by score
            return move_scores[0][1]  # Return best score
        
        all_moves = self.get_legal_moves()
        move_scores = []
        
        for move in all_moves:
            self.save_state()  # Save board state before simulating
            self.make_move(move)
            score = minimax(self, depth - 1, maximizing_player=(self.turn == "white"))
            move_scores.append((move, score))
            self.restore_state()  # Undo move
        
        move_scores.sort(key=lambda x: x[1], reverse=self.turn == "white")  # White prefers max, black prefers min
        best_moves = move_scores[:width]  # Keep top `width` moves
        
        return best_moves


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

    def update_game_state(self, win, tree_width, tree_depth):
        """Updates game state when a move is made"""
        self.board.update_heatmap()  # Recalculate influence heatmap
        self.board.update_legal_moves()  # Compute available & legal moves
        self.board.predictions = self.board.predict(tree_width, tree_depth)  # Generate predictions
        self.sidebar.update_display(self.board.heatmap, self.board.predictions)  # Send to UI

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
            game.update_game_state(win, game.sidebar.tree_width, game.sidebar.tree_depth)
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

class Board:
    def __init__(self):
        self.bitboards = self.init_bitboards()  # Store board state in bitboards
        self.heatmap = np.zeros((8, 8), dtype=int)  # Heatmap for influence
        self.fen_log = []  # Stores FEN for debugging & history
        self.predictions = []  # Store evaluated moves


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

    def generate_moves(self):
        """Use bitwise operations to generate legal moves"""
        # Example: White pawn moves (shift bitboard)
        white_pawns = self.bitboards['P']
        pawn_moves = (white_pawns >> 8) & ~self.occupied  # Move forward
        return pawn_moves
    
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
                new_board = apply_move(board_state, (piece, move))

                # Check if king is in check
                _, enemy_influence = get_available_moves(new_board, enemy_color)
                if not (enemy_influence['king'] & king_position):
                    legal_moves[piece] |= move  # Add move if legal

        return legal_moves


#Board Update
    def update(self, move):
        """Handles a move and updates available moves."""
        # Apply the move
        self.board_state = self.apply_move(self.board_state, move)

        # Switch turn
        self.current_turn = "black" if self.current_turn == "white" else "white"

        # Recompute available moves for the new player
        self.available_moves, self.influence = get_available_moves(self.board_state, self.current_turn)

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

    def calculate_global_heatmap(self):
        """Computes a score by summing influence of both colors."""
        white_control = self.calculate_control("white")
        black_control = self.calculate_control("black")
        return sum(white_control) - sum(black_control)


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
        
        # Move Recommendations
        self.best_moves = []
    
    def draw(self, screen, turn, predictions):
        pygame.draw.rect(screen, self.bg_color, self.rect)
        
        # Display current turn
        turn_surf = self.font.render(f"Turn: {turn}", True, self.text_color)
        screen.blit(turn_surf, (self.title_area.x, self.title_area.y + 30))

        # Display Move Predictions
        self.best_moves = predictions  # Store predictions
        for i, move in enumerate(self.best_moves[:5]):  # Show top 5
            move_surf = self.small_font.render(
                f"{move[0]} -> {move[1]} ({move[3]:.1f})", True, self.text_color
            )
            screen.blit(move_surf, (self.moves_area.x, self.moves_area.y + i * 20))
        
        # Method Selection
        for i, method in enumerate(self.methods):
            color = self.highlight_color if i == self.selected_method else self.text_color
            method_surf = self.small_font.render(method, True, color)
            screen.blit(method_surf, (self.method_area.x, self.method_area.y + i * 25))
        
        # Input Box
        pygame.draw.rect(screen, (200, 200, 200), self.input_box, 2)
        input_surf = self.small_font.render(self.input_text, True, self.text_color)
        screen.blit(input_surf, (self.input_box.x + 5, self.input_box.y + 5))
    
    def update_moves(self, moves):
        """Updates the sidebar with the best predicted moves."""
        self.best_moves = [(move[0], move[1], move[2], move[3]) for move in moves]  # (Piece, From, To, Score)

    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.input_box.collidepoint(event.pos):
                self.input_active = not self.input_active
            else:
                self.input_active = False
        elif event.type == pygame.KEYDOWN and self.input_active:
            if event.key == pygame.K_RETURN:
                self.input_text = ''
            elif event.key == pygame.K_BACKSPACE:
                self.input_text = self.input_text[:-1]
            else:
                self.input_text += event.unicode
